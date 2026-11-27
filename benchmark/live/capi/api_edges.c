/* SPDX-License-Identifier: Apache-2.0 */
/* libreflop edge cases (0.1.6 changes) and a latency benchmark. Prints one JSON line per check.
 *   api_edges <device> <bench_iters>
 * Checks: fetch size query (buf=NULL, cap=0), ERANGE on a short buffer (nothing copied), nbytes=NULL rejected,
 * upload nbytes mismatch rejected, register with an open node rejected, unknown op rejected, bad server name,
 * nested nodes, then times upload / execute / fetch round trips. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "reflop.h"

static int failures = 0;

static void report(const char *name, int ok, const char *detail) {
  char buf[400];
  size_t j = 0;
  for (const char *p = detail ? detail : ""; *p && j < sizeof buf - 2; p++) {
    if (*p == '"' || *p == '\\') buf[j++] = '\'';
    else if ((unsigned char)*p >= 32) buf[j++] = *p;
  }
  buf[j] = 0;
  printf("{\"check\":\"%s\",\"ok\":%s,\"detail\":\"%s\"}\n", name, ok ? "true" : "false", buf);
  if (!ok) failures++;
}

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  const char *device = argc > 1 ? argv[1] : "auto";
  int iters = argc > 2 ? atoi(argv[2]) : 50;
  reflop_session s;
  if (reflop_connect(NULL, device, &s) != 0) {
    report("connect", 0, reflop_last_error());
    return 1;
  }
  report("connect", 1, reflop_version());

  reflop_session bad;
  int rc = reflop_connect("no-such-server-e2e", device, &bad);
  report("bad_server_name_rejected", rc != 0, reflop_last_error());

  float data[64];
  for (int i = 0; i < 64; i++) data[i] = (float)i;
  int64_t shp[2] = {8, 8};
  reflop_handle h;
  rc = reflop_upload(s, REFLOP_F32, 2, shp, data, sizeof data - 4, &h);
  report("upload_nbytes_mismatch_rejected", rc != 0, reflop_last_error());
  rc = reflop_upload(s, REFLOP_F32, 2, shp, data, sizeof data, &h);
  report("upload", rc == 0, rc ? reflop_last_error() : "");

  size_t n = 0;
  rc = reflop_fetch(s, h, NULL, 0, &n);
  char msg[200];
  snprintf(msg, sizeof msg, "rc=%d nbytes=%zu", rc, n);
  report("fetch_size_query", rc == 0 && n == sizeof data, msg);

  float small[16];
  for (int i = 0; i < 16; i++) small[i] = -7.0f;
  n = 0;
  rc = reflop_fetch(s, h, small, sizeof small, &n);
  int untouched = 1;
  for (int i = 0; i < 16; i++) untouched &= small[i] == -7.0f;
  snprintf(msg, sizeof msg, "rc=%d (ERANGE=%d) nbytes=%zu untouched=%d err=%s", rc, REFLOP_ERANGE, n, untouched,
           reflop_last_error());
  report("fetch_short_buffer_erange", rc == REFLOP_ERANGE && untouched && n == sizeof data, msg);

  rc = reflop_fetch(s, h, small, sizeof small, NULL);
  report("fetch_null_nbytes_rejected", rc != 0, reflop_last_error());

  float back[64];
  rc = reflop_fetch(s, h, back, sizeof back, &n);
  report("fetch_roundtrip", rc == 0 && memcmp(back, data, sizeof data) == 0, rc ? reflop_last_error() : "");

  /* register with a node still open */
  reflop_graph g = reflop_graph_new();
  reflop_value x = reflop_graph_input(g, REFLOP_F32, 2, shp);
  reflop_op_begin(g, "aten.relu.default");
  reflop_arg_tensor(g, x);
  reflop_graph_id gid;
  rc = reflop_register(s, g, &gid);
  report("register_open_node_rejected", rc != 0, reflop_last_error());
  reflop_graph_free(g);

  /* unknown operator */
  g = reflop_graph_new();
  x = reflop_graph_input(g, REFLOP_F32, 2, shp);
  reflop_value y;
  reflop_op_begin(g, "aten.definitely_not_an_op.default");
  reflop_arg_tensor(g, x);
  int erc = reflop_op_end(g, &y);
  rc = erc == 0 ? reflop_graph_output(g, y, REFLOP_F32, 2, shp) : erc;
  if (rc == 0) rc = reflop_register(s, g, &gid);
  report("unknown_op_rejected", rc != 0, reflop_last_error());
  reflop_graph_free(g);

  /* nested nodes: mm(x, t(x)) built inline */
  g = reflop_graph_new();
  x = reflop_graph_input(g, REFLOP_F32, 2, shp);
  reflop_value inner, z;
  int ok = reflop_op_begin(g, "aten.mm.default") == 0 && reflop_arg_tensor(g, x) == 0 &&
           reflop_op_begin(g, "aten.t.default") == 0 && reflop_arg_tensor(g, x) == 0 &&
           reflop_op_end(g, &inner) == 0 && reflop_arg_tensor(g, inner) == 0 && reflop_op_end(g, &z) == 0 &&
           reflop_graph_output(g, z, REFLOP_F32, 2, shp) == 0 && reflop_register(s, g, &gid) == 0;
  reflop_handle out;
  if (ok) ok = reflop_execute(s, gid, &h, 1, &out, 1) == 0 && reflop_fetch(s, out, back, sizeof back, &n) == 0;
  double err = 0;
  for (int i = 0; ok && i < 8; i++)
    for (int j = 0; j < 8; j++) {
      double acc = 0;
      for (int k = 0; k < 8; k++) acc += data[i * 8 + k] * data[j * 8 + k];
      err = fmax(err, fabs(acc - back[i * 8 + j]));
    }
  snprintf(msg, sizeof msg, "max err %.3g %s", err, ok ? "" : reflop_last_error());
  report("nested_nodes_execute", ok && err < 1e-3, msg);

  /* latency benchmark: execute + fetch of a 64-float result, and 1 MiB upload */
  if (ok && iters > 0) {
    double t0 = now();
    for (int i = 0; i < iters; i++) {
      reflop_handle o2;
      reflop_execute(s, gid, &h, 1, &o2, 1);
      reflop_fetch(s, o2, back, sizeof back, &n);
      reflop_free(s, o2);
    }
    double t_exec = (now() - t0) / iters;
    static float mb[262144];
    int64_t s1[1] = {262144};
    t0 = now();
    for (int i = 0; i < iters / 5 + 1; i++) {
      reflop_handle hu;
      reflop_upload(s, REFLOP_F32, 1, s1, mb, sizeof mb, &hu);
      size_t q;
      reflop_fetch(s, hu, NULL, 0, &q);
      reflop_free(s, hu);
    }
    double t_up = (now() - t0) / (iters / 5 + 1);
    printf("{\"bench\":\"c_api\",\"device\":\"%s\",\"execute_fetch_ms\":%.3f,\"upload_1mib_ms\":%.3f,\"iters\":%d}\n",
           device, t_exec * 1e3, t_up * 1e3, iters);
  }
  reflop_graph_free(g);
  reflop_free(s, h);
  rc = reflop_close(s);
  report("close", rc == 0, rc ? reflop_last_error() : "");
  return failures ? 2 : 0;
}
