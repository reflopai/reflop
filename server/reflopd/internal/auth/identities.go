// SPDX-License-Identifier: Apache-2.0

package auth

import (
	"crypto/ed25519"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base32"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/sys/unix"

	pb "github.com/reflopai/reflop/pkg/reflopv1"
	"github.com/reflopai/reflop/pkg/wire"
	"github.com/reflopai/reflop/server/reflopd/internal/config"
	"github.com/reflopai/reflop/server/reflopd/internal/gpu"
)

// NameRe constrains identity names.
var NameRe = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]{0,31}$`)

// ScopedName gives identities with the same Guest name independent state per GPU.
func ScopedName(name, uuid string) string {
	if uuid == "" {
		return name
	}
	return name + "@" + uuid
}

// OfGuest says whether the identity named full belongs to the Guest name:
// full is name itself (paired before 0.2.10) or name@<gpu-uuid>, which
// `reflopd users` lists as name. NameRe keeps "@" out of names.
func OfGuest(full, name string) bool {
	if full == name {
		return true
	}
	base, uuid, scoped := strings.Cut(full, "@")
	return scoped && base == name && strings.HasPrefix(uuid, "gpu-")
}

// Key is one registered public key.
type Key struct {
	Public  []byte    `json:"public"` // 32-byte Ed25519
	Added   time.Time `json:"added"`
	Via     string    `json:"via"` // "tls" or "ssh"
	Comment string    `json:"comment,omitempty"`
}

// Identity is one paired Guest.
type Identity struct {
	ID      string    `json:"id"` // 32 hex chars
	Name    string    `json:"name"`
	GPUUUID string    `json:"gpu_uuid,omitempty"`
	Keys    []Key     `json:"keys"`
	Created time.Time `json:"created"`
	Revoked bool      `json:"revoked,omitempty"`
	// GuestUnpaired is when the Guest ran `reflop unpair`, which revokes
	// the identity; the Owner is asked to delete it (ADR-025).
	GuestUnpaired *time.Time `json:"guest_unpaired,omitempty"`
}

// IDBytes returns the 16-byte identity id.
func (i *Identity) IDBytes() []byte { b, _ := hex.DecodeString(i.ID); return b }

// Invite is a pending one-time pairing secret.
type Invite struct {
	ID      string    `json:"id"`     // 32 hex
	Secret  []byte    `json:"secret"` // 32 bytes
	Name    string    `json:"name"`
	GPUUUID string    `json:"gpu_uuid,omitempty"`
	Expires time.Time `json:"expires"`
	Used    bool      `json:"used,omitempty"`
}

type db struct {
	Identities []*Identity `json:"identities"`
	Invites    []*Invite   `json:"invites"`
}

// Store persists identities and invites in one 0600 JSON file. All
// methods are safe for concurrent use; another reflopd command (e.g.
// `reflopd revoke`) changes the file and the daemon reloads on notice.
type Store struct {
	mu       sync.Mutex
	path     string
	db       db
	lastFile os.FileInfo
}

// Open loads (or initializes) the store.
func Open(dir string) (*Store, error) {
	s := &Store{path: filepath.Join(dir, "identities.json")}
	return s, s.Reload()
}

// Reload re-reads the file.
func (s *Store) Reload() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.reloadLocked()
}

// refreshLocked reloads when another reflopd command changed the file
// (invite, revoke, ssh key registration).
func (s *Store) refreshLocked() error {
	st, err := os.Stat(s.path)
	if err == nil && s.lastFile != nil && os.SameFile(st, s.lastFile) &&
		st.ModTime().Equal(s.lastFile.ModTime()) && st.Size() == s.lastFile.Size() {
		return nil
	}
	return s.reloadLocked()
}

func (s *Store) reloadLocked() error {
	f, err := os.Open(s.path)
	if errors.Is(err, os.ErrNotExist) {
		s.db = db{}
		s.lastFile = nil
		return nil
	}
	if err != nil {
		return err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return err
	}
	var d db
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&d); err != nil {
		return fmt.Errorf("%s: %w", s.path, err)
	}
	for _, id := range d.Identities {
		if id == nil || !validScope(id.Name, id.GPUUUID) {
			return fmt.Errorf("%s: invalid identity GPU scope", s.path)
		}
	}
	for _, inv := range d.Invites {
		if inv == nil || !validScope(inv.Name, inv.GPUUUID) {
			return fmt.Errorf("%s: invalid invite GPU scope", s.path)
		}
	}
	s.db = d
	s.lastFile = st
	return nil
}

// lockWrite serializes read-modify-write operations with the daemon and
// other Owner CLI processes. The caller holds s.mu until the lock is closed.
func (s *Store) lockWrite() (*os.File, error) {
	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(filepath.Join(dir, "identities.lock"), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	for {
		err = unix.Flock(int(f.Fd()), unix.LOCK_EX)
		if !errors.Is(err, unix.EINTR) {
			break
		}
	}
	if err == nil {
		err = s.refreshLocked()
	}
	if err != nil {
		f.Close()
		return nil, err
	}
	return f, nil
}

func validScope(name, uuid string) bool {
	if uuid == "" {
		return NameRe.MatchString(name)
	}
	if (gpu.GPU{UUID: uuid}).StableUUID() != uuid {
		return false
	}
	base, ok := strings.CutSuffix(name, "@"+uuid)
	return ok && NameRe.MatchString(base)
}

func (s *Store) saveLocked() error {
	// Any failed save must discard the in-memory write before the next read.
	s.lastFile = nil
	b, err := json.MarshalIndent(&s.db, "", "  ")
	if err != nil {
		return err
	}
	if err := config.WriteFileAtomic(s.path, append(b, '\n'), 0o600); err != nil {
		return err
	}
	s.lastFile, _ = os.Stat(s.path)
	return nil
}

func randHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	return hex.EncodeToString(b)
}

// CreateInvite records a one-time invite for name.
func (s *Store) CreateInvite(name string, ttl time.Duration) (*Invite, error) {
	return s.CreateGPUInvite(name, ttl, "")
}

// CreateGPUInvite records the Owner-selected scope with the invite secret.
func (s *Store) CreateGPUInvite(name string, ttl time.Duration, uuid string) (*Invite, error) {
	if !NameRe.MatchString(name) {
		return nil, fmt.Errorf("invalid name %q: use 1-32 of a-z 0-9 _ - (starting with a letter or digit)", name)
	}
	name = ScopedName(name, uuid)
	if !validScope(name, uuid) {
		return nil, errors.New("invalid GPU UUID")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	lock, err := s.lockWrite()
	if err != nil {
		return nil, err
	}
	defer lock.Close()
	for _, id := range s.db.Identities {
		if id.Name == name && id.Revoked {
			return nil, fmt.Errorf("identity %q was revoked; delete it first with `reflopd unpair %s` "+
				"(or `reflopd users delete %s`), or choose another name", name, name, name)
		}
	}
	secret := make([]byte, 32)
	if _, err := rand.Read(secret); err != nil {
		return nil, err
	}
	inv := &Invite{ID: randHex(16), Secret: secret, Name: name, GPUUUID: uuid, Expires: time.Now().Add(ttl).UTC().Truncate(time.Second)}
	s.pruneLocked()
	s.db.Invites = append(s.db.Invites, inv)
	return inv, s.saveLocked()
}

func (s *Store) pruneLocked() {
	now := time.Now()
	kept := s.db.Invites[:0]
	for _, inv := range s.db.Invites {
		if !inv.Used && now.Before(inv.Expires) {
			kept = append(kept, inv)
		}
	}
	s.db.Invites = kept
}

// ErrInvite covers every invite failure; callers must not reveal which.
var ErrInvite = errors.New("invalid, expired or already used invite")

// InviteError is ErrInvite with the actual reason. The reason is for the
// Owner only (daemon log, audit trail); the Guest is sent ErrInvite's text.
type InviteError struct{ Reason string }

func (e *InviteError) Error() string        { return ErrInvite.Error() + ": " + e.Reason }
func (e *InviteError) Is(target error) bool { return target == ErrInvite }

func inviteErr(format string, args ...any) error {
	return &InviteError{Reason: fmt.Sprintf(format, args...)}
}

// Redeem verifies a Pair request and registers the key. The invite is
// consumed on the first attempt that names it, successful or not, so a
// guessed or stolen invite id cannot be brute-forced.
func (s *Store) Redeem(req *pb.Pair, nonce, serverFP []byte) (*Identity, error) {
	if len(req.InviteId) != 16 || len(req.PublicKey) != ed25519.PublicKeySize || len(req.Mac) != 32 || len(req.Signature) != ed25519.SignatureSize {
		return nil, inviteErr("malformed pair request")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	lock, err := s.lockWrite()
	if err != nil {
		return nil, err
	}
	defer lock.Close()
	var inv *Invite
	idHex := hex.EncodeToString(req.InviteId)
	for _, x := range s.db.Invites {
		if subtle.ConstantTimeCompare([]byte(x.ID), []byte(idHex)) == 1 {
			inv = x
		}
	}
	now := time.Now()
	switch {
	case inv == nil:
		return nil, inviteErr("no such invite: not created by this server, or used or expired and since removed")
	case inv.Used:
		return nil, inviteErr("the invite for %q was already used (any earlier pairing attempt uses it up, even a failed one)", inv.Name)
	case now.After(inv.Expires):
		return nil, inviteErr("the invite for %q expired at %s, %s ago", inv.Name,
			inv.Expires.UTC().Format(time.RFC3339), now.Sub(inv.Expires).Round(time.Second))
	}
	inv.Used = true
	msg := PairMessage(nonce, serverFP, req.InviteId, req.PublicKey)
	mac := hmac.New(sha256.New, inv.Secret)
	mac.Write(msg)
	if !hmac.Equal(mac.Sum(nil), req.Mac) || !ed25519.Verify(req.PublicKey, msg, req.Signature) {
		_ = s.saveLocked()
		return nil, inviteErr("wrong secret or signature for the invite for %q; the invite is now used up", inv.Name)
	}
	if req.GpuUuid != inv.GPUUUID {
		_ = s.saveLocked()
		return nil, inviteErr("GPU scope does not match the invite for %q", inv.Name)
	}
	if owner := s.byKeyLocked(req.PublicKey); owner != nil {
		_ = s.saveLocked()
		return nil, &KeyRegisteredError{Name: owner.Name}
	}
	id := s.byNameLocked(inv.Name)
	if id == nil {
		id = &Identity{ID: randHex(16), Name: inv.Name, GPUUUID: inv.GPUUUID, Created: time.Now().UTC().Truncate(time.Second)}
		s.db.Identities = append(s.db.Identities, id)
	}
	id.Keys = append(id.Keys, Key{Public: req.PublicKey, Added: time.Now().UTC().Truncate(time.Second), Via: "tls"})
	s.pruneLocked()
	return id, s.saveLocked()
}

// KeyRegisteredError refuses a Pair whose key already belongs to an active
// identity. The Guest has forgotten this server (`reflop unpair` while it
// could not reach it, or an older `reflop servers remove`) but the Owner
// has not unpaired it yet. The message is for the Guest: it passed the
// invite's MAC, and the identity is its own.
type KeyRegisteredError struct{ Name string }

func (e *KeyRegisteredError) Error() string {
	return fmt.Sprintf("this machine's key is still registered to %q on this server. Ask the Owner to run "+
		"`reflopd unpair %s`, then pair again with a new invite", e.Name, e.Name)
}

// AddKey registers a key directly (SSH pipe mode: the Owner pastes the
// Guest's public key).
func (s *Store) AddKey(name string, pub []byte, via, comment string) (*Identity, error) {
	return s.AddGPUKey(name, pub, via, comment, "")
}

// AddGPUKey registers an SSH key for the selected GPU.
func (s *Store) AddGPUKey(name string, pub []byte, via, comment, uuid string) (*Identity, error) {
	if !NameRe.MatchString(name) {
		return nil, fmt.Errorf("invalid name %q", name)
	}
	name = ScopedName(name, uuid)
	if !validScope(name, uuid) {
		return nil, errors.New("invalid GPU UUID")
	}
	if len(pub) != ed25519.PublicKeySize {
		return nil, errors.New("public key must be Ed25519")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	lock, err := s.lockWrite()
	if err != nil {
		return nil, err
	}
	defer lock.Close()
	if owner := s.byKeyLocked(pub); owner != nil {
		if owner.Name == name {
			return owner, nil
		}
		return nil, fmt.Errorf("this key is already registered to %q; to give it to %q, run `reflopd unpair %s` first", owner.Name, name, owner.Name)
	}
	id := s.byNameLocked(name)
	if id != nil && id.Revoked {
		return nil, fmt.Errorf("identity %q was revoked; delete it first with `reflopd unpair %s` "+
			"(or `reflopd users delete %s`), or choose another name", name, name, name)
	}
	if id == nil {
		id = &Identity{ID: randHex(16), Name: name, GPUUUID: uuid, Created: time.Now().UTC().Truncate(time.Second)}
		s.db.Identities = append(s.db.Identities, id)
	}
	id.Keys = append(id.Keys, Key{Public: pub, Added: time.Now().UTC().Truncate(time.Second), Via: via, Comment: comment})
	return id, s.saveLocked()
}

func (s *Store) byKeyLocked(pub []byte) *Identity {
	for _, id := range s.db.Identities {
		for _, k := range id.Keys {
			if subtle.ConstantTimeCompare(k.Public, pub) == 1 {
				return id
			}
		}
	}
	return nil
}

func (s *Store) byNameLocked(name string) *Identity {
	for _, id := range s.db.Identities {
		if id.Name == name {
			return id
		}
	}
	return nil
}

// ErrAuth covers every authentication failure.
var ErrAuth = errors.New("authentication failed")

// Authenticate checks a signature and returns the key's identity. If
// wantID is non-empty (SSH mode), the key must belong to that identity.
func (s *Store) Authenticate(req *pb.Authenticate, nonce, serverFP []byte, wantID string) (*Identity, error) {
	if len(req.PublicKey) != ed25519.PublicKeySize || len(req.Signature) != ed25519.SignatureSize {
		return nil, ErrAuth
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.refreshLocked(); err != nil {
		return nil, ErrAuth
	}
	id := s.byKeyLocked(req.PublicKey)
	if id == nil || id.Revoked {
		return nil, ErrAuth
	}
	if wantID != "" && id.ID != wantID {
		return nil, ErrAuth
	}
	if !ed25519.Verify(req.PublicKey, AuthMessage(nonce, serverFP, req.PublicKey), req.Signature) {
		return nil, ErrAuth
	}
	c := *id
	return &c, nil
}

// Revoke marks an identity revoked and drops its keys.
func (s *Store) Revoke(name string) (*Identity, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	lock, err := s.lockWrite()
	if err != nil {
		return nil, err
	}
	defer lock.Close()
	id := s.byNameLocked(name)
	if id == nil {
		return nil, fmt.Errorf("no identity named %q", name)
	}
	removed := s.revokeLocked(id)
	return &removed, s.saveLocked()
}

// revokeLocked marks id revoked, drops its keys and uses up its invites. It
// returns the identity as it was, keys included.
func (s *Store) revokeLocked(id *Identity) Identity {
	id.Revoked = true
	removed := *id
	id.Keys = nil
	for i, inv := range s.db.Invites {
		if inv.Name == id.Name {
			s.db.Invites[i].Used = true
		}
	}
	return removed
}

// GuestUnpair revokes the identity with id idHex at its own Guest's request
// (`reflop unpair`) and records when. Only an authenticated connection of
// that identity calls it, so a Guest can give up only its own access. The
// Owner still deletes the identity (ADR-025).
func (s *Store) GuestUnpair(idHex string) (*Identity, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	lock, err := s.lockWrite()
	if err != nil {
		return nil, err
	}
	defer lock.Close()
	for _, id := range s.db.Identities {
		if id.ID == idHex {
			if id.Revoked {
				return nil, ErrAuth
			}
			now := time.Now().UTC().Truncate(time.Second)
			id.GuestUnpaired = &now
			removed := s.revokeLocked(id)
			return &removed, s.saveLocked()
		}
	}
	return nil, ErrAuth
}

// ErrNotRevoked refuses to delete an identity that can still connect.
var ErrNotRevoked = errors.New("identity is active")

// Delete removes a revoked identity and its invites from the store, which
// frees the name for a new invite. Active identities are refused with
// ErrNotRevoked: revoke them first.
func (s *Store) Delete(name string) (*Identity, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	lock, err := s.lockWrite()
	if err != nil {
		return nil, err
	}
	defer lock.Close()
	idx := -1
	for i, id := range s.db.Identities {
		if id.Name == name {
			idx = i
		}
	}
	if idx < 0 {
		return nil, fmt.Errorf("no identity named %q", name)
	}
	removed := *s.db.Identities[idx]
	if !removed.Revoked {
		return nil, fmt.Errorf("%w: %q; revoke it first with `reflopd revoke %s`", ErrNotRevoked, name, name)
	}
	s.db.Identities = append(s.db.Identities[:idx], s.db.Identities[idx+1:]...)
	invites := s.db.Invites[:0]
	for _, inv := range s.db.Invites {
		if inv.Name != name {
			invites = append(invites, inv)
		}
	}
	s.db.Invites = invites
	return &removed, s.saveLocked()
}

// List returns all identities sorted by name.
func (s *Store) List() []Identity {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.refreshLocked(); err != nil {
		return nil
	}
	out := make([]Identity, 0, len(s.db.Identities))
	for _, id := range s.db.Identities {
		out = append(out, *id)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Get returns one identity by name.
func (s *Store) Get(name string) (Identity, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.refreshLocked(); err != nil {
		return Identity{}, false
	}
	if id := s.byNameLocked(name); id != nil {
		return *id, true
	}
	return Identity{}, false
}

// IsActive reports whether identity id exists and is not revoked.
func (s *Store) IsActive(idHex string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.refreshLocked(); err != nil {
		return false
	}
	for _, id := range s.db.Identities {
		if id.ID == idHex {
			return !id.Revoked
		}
	}
	return false
}

// InvitePrefix starts every invite code.
const InvitePrefix = "reflop1-"

var b32 = base32.StdEncoding.WithPadding(base32.NoPadding)

// EncodeInvite renders an invite code.
func EncodeInvite(c *pb.InviteCode) (string, error) {
	b, err := wire.Marshal(c)
	if err != nil {
		return "", err
	}
	return InvitePrefix + strings.ToLower(b32.EncodeToString(b)), nil
}

// DecodeInvite parses an invite code strictly.
func DecodeInvite(s string) (*pb.InviteCode, error) {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, InvitePrefix) {
		return nil, errors.New("not a reflop invite code")
	}
	raw, err := b32.DecodeString(strings.ToUpper(strings.TrimPrefix(s, InvitePrefix)))
	if err != nil {
		return nil, fmt.Errorf("invite code: %w", err)
	}
	c := &pb.InviteCode{}
	if err := wire.Unmarshal(raw, c); err != nil {
		return nil, err
	}
	return c, nil
}
