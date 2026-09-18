// Thin pybind11 wrappers around qft++ used as a numerical reference for qftppy tests.
//
// Tensor-valued functions return the flat qft++ element storage, in which the FIRST
// Lorentz index varies fastest (index = mu + 4*nu + 16*rho + ...). Convert to qftppy's
// layout on the Python side with tests/helpers.py:from_cpp. Matrix-valued functions
// are unpacked here and returned in qftppy's layout directly.
#include <complex>
#include <stdexcept>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "qft++.h"

namespace py = pybind11;
using cd = std::complex<double>;
using Events = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

Vector4<double> row(const py::detail::unchecked_reference<double, 2> &p, py::ssize_t i) {
  return Vector4<double>(p(i, 0), p(i, 1), p(i, 2), p(i, 3));
}

template <typename T>
py::array_t<T> flat_tensor(const Tensor<T> &t) {
  py::array_t<T> out(t.Size());
  auto o = out.template mutable_unchecked<1>();
  for (int k = 0; k < t.Size(); ++k) o(k) = t[k];
  return out;
}

void check_events(const Events &p) {
  if (p.ndim() != 2 || p.shape(1) != 4) throw std::invalid_argument("expected an (N, 4) array of four-momenta");
}

}  // namespace

py::array_t<double> metric() { return flat_tensor<double>(MetricTensor()); }

py::array_t<double> levi_civita() { return flat_tensor<double>(LeviCivitaTensor()); }

// (4 mu, 4, 4): gamma^mu as 4x4 matrices
py::array_t<cd> dirac_gamma() {
  DiracGamma g;
  py::array_t<cd> out({4, 4, 4});
  auto o = out.mutable_unchecked<3>();
  for (int mu = 0; mu < 4; ++mu) {
    Matrix<cd> g_mu = g(mu);
    for (int a = 0; a < 4; ++a)
      for (int b = 0; b < 4; ++b) o(mu, a, b) = g_mu(a, b);
  }
  return out;
}

py::array_t<double> dirac_gamma5() {
  DiracGamma5 g5;
  py::array_t<double> out({4, 4});
  auto o = out.mutable_unchecked<2>();
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 4; ++b) o(a, b) = g5(a, b);
  return out;
}

// (4 mu, 4 nu, 4, 4): sigma^{mu nu} as 4x4 matrices
py::array_t<cd> dirac_sigma() {
  DiracSigma s;
  py::array_t<cd> out({4, 4, 4, 4});
  auto o = out.mutable_unchecked<4>();
  for (int mu = 0; mu < 4; ++mu)
    for (int nu = 0; nu < 4; ++nu)
      for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) o(mu, nu, a, b) = s(a, b).Element(mu, nu);
  return out;
}

// (3, 2, 2): sigma^1, sigma^2, sigma^3 (qft++ component 0 is dropped)
py::array_t<cd> pauli_sigma() {
  PauliSigma s;
  py::array_t<cd> out({3, 2, 2});
  auto o = out.mutable_unchecked<3>();
  for (int i = 0; i < 3; ++i) {
    Matrix<cd> s_i = s(i + 1);
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b) o(i, a, b) = s_i(a, b);
  }
  return out;
}

// (N, 4^rank) real, flat qft++ layout
py::array_t<double> orbital_tensor(Events pa, Events pb, int rank) {
  check_events(pa);
  check_events(pb);
  auto a = pa.unchecked<2>();
  auto b = pb.unchecked<2>();
  const py::ssize_t n = a.shape(0);
  const py::ssize_t size = py::ssize_t(1) << (2 * rank);
  py::array_t<double> out({n, size});
  auto o = out.mutable_unchecked<2>();
  OrbitalTensor L(rank);
  for (py::ssize_t i = 0; i < n; ++i) {
    L.SetP4(row(a, i), row(b, i));
    for (py::ssize_t k = 0; k < size; ++k) o(i, k) = L[k];
  }
  return out;
}

// (N, 4^spin) complex, flat qft++ layout
py::array_t<cd> pol_vector(Events p4, double mass, int spin, int mz) {
  check_events(p4);
  auto p = p4.unchecked<2>();
  const py::ssize_t n = p.shape(0);
  const py::ssize_t size = py::ssize_t(1) << (2 * spin);
  py::array_t<cd> out({n, size});
  auto o = out.mutable_unchecked<2>();
  PolVector eps{Spin(spin)};
  for (py::ssize_t i = 0; i < n; ++i) {
    eps.SetP4(row(p, i), mass);
    const Tensor<cd> &t = eps(Spin(mz));
    for (py::ssize_t k = 0; k < size; ++k) o(i, k) = t[k];
  }
  return out;
}

// (N, 4^(2 spin)) complex, flat qft++ layout
py::array_t<cd> pol_projector(Events p4, double mass, int spin) {
  check_events(p4);
  auto p = p4.unchecked<2>();
  const py::ssize_t n = p.shape(0);
  const py::ssize_t size = py::ssize_t(1) << (4 * spin);
  py::array_t<cd> out({n, size});
  auto o = out.mutable_unchecked<2>();
  PolVector eps{Spin(spin)};
  for (py::ssize_t i = 0; i < n; ++i) {
    eps.SetP4(row(p, i), mass);
    const Tensor<cd> &t = eps.Projector();
    for (py::ssize_t k = 0; k < size; ++k) o(i, k) = t[k];
  }
  return out;
}

// (N, 4) complex spin-1/2 particle spinor u(p, mz)
py::array_t<cd> dirac_u(Events p4, double mass, double mz) {
  check_events(p4);
  auto p = p4.unchecked<2>();
  const py::ssize_t n = p.shape(0);
  py::array_t<cd> out({n, py::ssize_t(4)});
  auto o = out.mutable_unchecked<2>();
  DiracSpinor u{Spin(1, 2)};
  for (py::ssize_t i = 0; i < n; ++i) {
    u.SetP4(row(p, i), mass);
    const auto &s = u(Spin(mz));
    for (int a = 0; a < 4; ++a) o(i, a) = s(a, 0).Element();
  }
  return out;
}

// (N, 4) complex spin-1/2 antiparticle spinor v(p, mz)
py::array_t<cd> dirac_v(Events p4, double mass, double mz) {
  check_events(p4);
  auto p = p4.unchecked<2>();
  const py::ssize_t n = p.shape(0);
  py::array_t<cd> out({n, py::ssize_t(4)});
  auto o = out.mutable_unchecked<2>();
  DiracAntiSpinor v;
  for (py::ssize_t i = 0; i < n; ++i) {
    v.SetP4(row(p, i), mass);
    const auto &s = v(Spin(mz));
    for (int a = 0; a < 4; ++a) o(i, a) = s(a, 0);
  }
  return out;
}

// (N, 4, 4) complex spin-1/2 particle projector
py::array_t<cd> dirac_projector(Events p4, double mass) {
  check_events(p4);
  auto p = p4.unchecked<2>();
  const py::ssize_t n = p.shape(0);
  py::array_t<cd> out({n, py::ssize_t(4), py::ssize_t(4)});
  auto o = out.mutable_unchecked<3>();
  DiracSpinor u{Spin(1, 2)};
  for (py::ssize_t i = 0; i < n; ++i) {
    u.SetP4(row(p, i), mass);
    const auto &proj = u.Projector();
    for (int a = 0; a < 4; ++a)
      for (int b = 0; b < 4; ++b) o(i, a, b) = proj(a, b).Element();
  }
  return out;
}

// (N, 4, 4) complex spin-1/2 antiparticle projector
py::array_t<cd> dirac_anti_projector(Events p4, double mass) {
  check_events(p4);
  auto p = p4.unchecked<2>();
  const py::ssize_t n = p.shape(0);
  py::array_t<cd> out({n, py::ssize_t(4), py::ssize_t(4)});
  auto o = out.mutable_unchecked<3>();
  DiracAntiSpinor v;
  for (py::ssize_t i = 0; i < n; ++i) {
    v.SetP4(row(p, i), mass);
    const auto &proj = v.Projector();
    for (int a = 0; a < 4; ++a)
      for (int b = 0; b < 4; ++b) o(i, a, b) = proj(a, b);
  }
  return out;
}

double clebsch(double j1, double m1, double j2, double m2, double J, double M) {
  return Clebsch(Spin(j1), Spin(m1), Spin(j2), Spin(m2), Spin(J), Spin(M));
}

py::array_t<double> wigner_d(double j, double m, double n, Events beta) {
  auto b = beta.unchecked<1>();
  py::array_t<double> out(b.shape(0));
  auto o = out.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < b.shape(0); ++i) o(i) = Wigner_d(Spin(j), Spin(m), Spin(n), b(i));
  return out;
}

py::array_t<cd> regge_propagator(Events t, Events s, double a, double b, double spin, int sig, int exp_fact) {
  auto tt = t.unchecked<1>();
  auto ss = s.unchecked<1>();
  if (tt.shape(0) != ss.shape(0)) throw std::invalid_argument("t and s must have the same length");
  py::array_t<cd> out(tt.shape(0));
  auto o = out.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < tt.shape(0); ++i) o(i) = ReggePropagator(tt(i), ss(i), a, b, Spin(spin), sig, exp_fact);
  return out;
}

// S, P and D-wave spin amplitudes for V1 -> p1 p2, V2 -> p3 p4, P0 -> V1 V2 (as in the fitter cross-check)
py::array_t<double> spd_amplitudes(Events p1, Events p2, Events p3, Events p4) {
  for (const auto *p : {&p1, &p2, &p3, &p4}) check_events(*p);
  auto a = p1.unchecked<2>();
  auto b = p2.unchecked<2>();
  auto c = p3.unchecked<2>();
  auto d = p4.unchecked<2>();
  const py::ssize_t n = a.shape(0);
  py::array_t<double> out({n, py::ssize_t(3)});
  auto o = out.mutable_unchecked<2>();
  OrbitalTensor L1_V1(1), L1_V2(1), L1_P0(1), L2_P0(2);
  LeviCivitaTensor lct;
  for (py::ssize_t i = 0; i < n; ++i) {
    Vector4<double> v1 = row(a, i), v2 = row(b, i), v3 = row(c, i), v4 = row(d, i);
    Vector4<double> pV1 = v1 + v2, pV2 = v3 + v4, p0 = pV1 + pV2;
    L1_V1.SetP4(v1, v2);
    L1_V2.SetP4(v3, v4);
    L1_P0.SetP4(pV1, pV2);
    L2_P0.SetP4(pV1, pV2);
    o(i, 0) = Tensor<double>(L1_V1 * L1_V2).Element();
    o(i, 1) = Tensor<double>((((lct * L1_P0) * L1_V1) * L1_V2) * p0).Element();
    o(i, 2) = Tensor<double>((L2_P0 * L1_V1) * L1_V2).Element();
  }
  return out;
}

PYBIND11_MODULE(qftcpp_ref, m) {
  m.doc() = "qft++ reference values for qftppy tests";
  m.def("metric", &metric);
  m.def("levi_civita", &levi_civita);
  m.def("dirac_gamma", &dirac_gamma);
  m.def("dirac_gamma5", &dirac_gamma5);
  m.def("dirac_sigma", &dirac_sigma);
  m.def("pauli_sigma", &pauli_sigma);
  m.def("orbital_tensor", &orbital_tensor, py::arg("pa"), py::arg("pb"), py::arg("rank"));
  m.def("pol_vector", &pol_vector, py::arg("p4"), py::arg("mass"), py::arg("spin"), py::arg("mz"));
  m.def("pol_projector", &pol_projector, py::arg("p4"), py::arg("mass"), py::arg("spin"));
  m.def("dirac_u", &dirac_u, py::arg("p4"), py::arg("mass"), py::arg("mz"));
  m.def("dirac_v", &dirac_v, py::arg("p4"), py::arg("mass"), py::arg("mz"));
  m.def("dirac_projector", &dirac_projector, py::arg("p4"), py::arg("mass"));
  m.def("dirac_anti_projector", &dirac_anti_projector, py::arg("p4"), py::arg("mass"));
  m.def("clebsch", &clebsch);
  m.def("wigner_d", &wigner_d, py::arg("j"), py::arg("m"), py::arg("n"), py::arg("beta"));
  m.def("regge_propagator", &regge_propagator, py::arg("t"), py::arg("s"), py::arg("a"), py::arg("b"),
        py::arg("spin"), py::arg("sig"), py::arg("exp_fact") = 1);
  m.def("spd_amplitudes", &spd_amplitudes);
}
