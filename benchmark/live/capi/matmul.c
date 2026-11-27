/* Multiply two matrices on the reflop server with libreflop; compare with a C reference.
 * Prints one JSON line. Exit 0 if max abs error <= 1e-4 * max(1, max|ref|). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "reflop.h"

#define CHECK(x) do { if ((x) != 0) { fail(#x); } } while (0)

static void fail(const char *what) {
  printf("{\"scenario\":\"S12_c_matmul\",\"verdict\":\"fail\",\"call\":\"%s\",\"error\":\"%s\"}\n", what,
         reflop_last_error());
  exit(1);
}

int main(int argc, char **argv) {
  const char *device = argc > 1 ? argv[1] : "auto";
  enum { M = 128, K = 96, N = 80 };
  static float a[M * K], b[K * N], c[M * N], ref[M * N];
  unsigned s = 12345;
  for (int i = 0; i < M * K; i++) { s = s * 1103515245u + 12345u; a[i] = (float)((s >> 8) % 2001) / 1000.0f - 1.0f; }
  for (int i = 0; i < K * N; i++) { s = s * 1103515245u + 12345u; b[i] = (float)((s >> 8) % 2001) / 1000.0f - 1.0f; }
  double scale = 1.0;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      double acc = 0;
      for (int k = 0; k < K; k++) acc += (double)a[i * K + k] * b[k * N + j];
      ref[i * N + j] = (float)acc;
      if (fabs(acc) > scale) scale = fabs(acc);
    }

  reflop_session sess;
  CHECK(reflop_connect(NULL, device, &sess));
  int64_t sa[2] = {M, K}, sb[2] = {K, N}, sc[2] = {M, N};
  reflop_handle ha, hb;
  CHECK(reflop_upload(sess, REFLOP_F32, 2, sa, a, sizeof a, &ha));
  CHECK(reflop_upload(sess, REFLOP_F32, 2, sb, b, sizeof b, &hb));

  reflop_graph g = reflop_graph_new();
  reflop_value x = reflop_graph_input(g, REFLOP_F32, 2, sa);
  reflop_value y = reflop_graph_input(g, REFLOP_F32, 2, sb);
  reflop_value z;
  CHECK(reflop_op_begin(g, "aten.mm.default"));
  CHECK(reflop_arg_tensor(g, x));
  CHECK(reflop_arg_tensor(g, y));
  CHECK(reflop_op_end(g, &z));
  CHECK(reflop_graph_output(g, z, REFLOP_F32, 2, sc));

  reflop_graph_id id;
  CHECK(reflop_register(sess, g, &id));
  reflop_handle in[2] = {ha, hb}, out[1];
  CHECK(reflop_execute(sess, id, in, 2, out, 1));
  size_t n = 0;
  memset(c, 0, sizeof c);
  CHECK(reflop_fetch(sess, out[0], c, sizeof c, &n));

  double err = 0;
  for (int i = 0; i < M * N; i++) err = fmax(err, fabs(c[i] - ref[i]));
  double rel = err / scale;
  int ok = rel <= 1e-4 && n == sizeof c;
  printf("{\"scenario\":\"S12_c_matmul\",\"verdict\":\"%s\",\"device\":\"%s\",\"shape\":[%d,%d,%d],"
         "\"bytes_back\":%zu,\"max_abs_err\":%.3e,\"max_rel_err\":%.3e,\"version\":\"%s\"}\n",
         ok ? "pass" : "fail", device, M, K, N, n, err, rel, reflop_version());
  reflop_free(sess, ha);
  reflop_free(sess, hb);
  reflop_free(sess, out[0]);
  reflop_graph_free(g);
  CHECK(reflop_close(sess));
  return ok ? 0 : 2;
}
