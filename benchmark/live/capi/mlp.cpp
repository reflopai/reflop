/* SPDX-License-Identifier: Apache-2.0 */
// Two-layer MLP forward (with GELU) on the reflop server through reflop.hpp; compare with a local reference.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "reflop.hpp"

int main(int argc, char **argv) {
  const int64_t B = 32, D = 64, H = 128, O = 10;
  auto fill = [](size_t n, float scale, float phase) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = scale * std::sin(0.37f * static_cast<float>(i) + phase);
    return v;
  };
  auto x = fill(B * D, 1.0f, 1.0f), w1 = fill(H * D, 0.15f, 2.0f), b1 = fill(H, 0.1f, 3.0f),
       w2 = fill(O * H, 0.1f, 4.0f), b2 = fill(O, 0.1f, 5.0f);

  // y = gelu(x W1^T + b1) W2^T + b2   (exact erf GELU)
  std::vector<double> h(B * H);
  std::vector<float> ref(B * O);
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < H; ++j) {
      double s = b1[j];
      for (int k = 0; k < D; ++k) s += static_cast<double>(x[i * D + k]) * w1[j * D + k];
      h[i * H + j] = 0.5 * s * (1.0 + std::erf(s / std::sqrt(2.0)));
    }
    for (int j = 0; j < O; ++j) {
      double s = b2[j];
      for (int k = 0; k < H; ++k) s += h[i * H + k] * w2[j * H + k];
      ref[i * O + j] = static_cast<float>(s);
    }
  }

  std::string device = argc > 1 ? argv[1] : "auto";
  try {
    reflop::Session s(nullptr, device.c_str());
    reflop::Graph g;
    auto gx = g.input(REFLOP_F32, {B, D});
    auto gw1 = g.input(REFLOP_F32, {H, D});
    auto gb1 = g.input(REFLOP_F32, {H});
    auto gw2 = g.input(REFLOP_F32, {O, H});
    auto gb2 = g.input(REFLOP_F32, {O});
    auto a1 = g.op("aten.addmm.default").tensor(gb1).tensor(gx).tensor(g.op("aten.t.default").tensor(gw1).end())
                  .i(1).i(1).end();
    auto r1 = g.op("aten.gelu.default").tensor(a1).enum_("none").end();
    auto y = g.op("aten.addmm.default").tensor(gb2).tensor(r1).tensor(g.op("aten.t.default").tensor(gw2).end())
                 .i(1).i(1).end();
    g.output(y, REFLOP_F32, {B, O});
    auto out = s.run(g, {s.upload(x, {B, D}), s.upload(w1, {H, D}), s.upload(b1, {H}), s.upload(w2, {O, H}),
                         s.upload(b2, {O})});
    auto got = s.fetch<float>(out[0]);
    double err = 0;
    for (size_t i = 0; i < ref.size(); ++i) err = std::max(err, static_cast<double>(std::fabs(got[i] - ref[i])));
    bool ok = got.size() == ref.size() && err <= 1e-4;
    std::printf("{\"scenario\":\"S12_cpp_mlp\",\"verdict\":\"%s\",\"device\":\"%s\",\"shape\":[%lld,%lld,%lld,%lld],"
                "\"max_abs_err\":%.3e}\n", ok ? "pass" : "fail", device.c_str(), static_cast<long long>(B),
                static_cast<long long>(D), static_cast<long long>(H), static_cast<long long>(O), err);
    return ok ? 0 : 2;
  } catch (const reflop::Error &e) {
    std::printf("{\"scenario\":\"S12_cpp_mlp\",\"verdict\":\"fail\",\"error\":\"%s\"}\n", e.what());
    return 1;
  }
}
