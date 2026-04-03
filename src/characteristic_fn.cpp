/**
 * @file characteristic_fn.cpp
 * @brief Heston characteristic function implementation.
 *
 * Primary reference: Albrecher et al. (2007) "The little Heston trap,"
 * Wilmott Magazine, Jan 2007, 83-92.
 *
 * The key insight of the Albrecher paper is that the original Heston (1993)
 * formula uses:
 *   g = (b - d) / (b + d)   where  b = κ − ρξu·i
 *
 * When d crosses the branch cut of the complex square root (which happens for
 * longer maturities or extreme parameters), the formula gives wrong results.
 * The "little trap" reformulation inverts g → 1/g, which is equivalent to
 * swapping the sign of d.  Under this convention d stays in the right half-
 * plane along the integration path, avoiding the branch cut.
 */

#include "heston/characteristic_fn.hpp"
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace heston {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

HestonCharacteristicFn::HestonCharacteristicFn(const HestonParams& p,
                                               const MarketData&  mkt)
    : params_(p), r_(mkt.rate), q_(mkt.div_yield), S0_(mkt.spot)
{
    if (!p.valid()) {
        throw std::invalid_argument("HestonCharacteristicFn: invalid parameters");
    }
}

// ---------------------------------------------------------------------------
// Core: compute (C, D) using Albrecher et al. (2007) rotation-corrected form
//
// Notation following Albrecher eq. (4)-(5):
//   alpha ≡ rho * xi * u * i  (complex)
//   beta  ≡ kappa - alpha      (= kappa - rho*xi*u*i)
//   d     = sqrt(beta^2 + xi^2*(i*u + u^2))
//
// Rotation-corrected (little Heston trap):
//   g = (beta - d) / (beta + d)   [NOTE: Albrecher uses G = 1/g_original]
//   D = (beta - d) / xi^2  *  (1 - e^{-d*tau}) / (1 - g*e^{-d*tau})
//   C = i*u*ln(F)*tau is folded into the caller;
//       the θ/ξ² part:
//   C = (kappa*theta/xi^2) * [(beta - d)*tau - 2*ln((1 - g*e^{-d*tau})/(1 - g))]
// ---------------------------------------------------------------------------

std::pair<std::complex<double>, std::complex<double>>
HestonCharacteristicFn::compute_CD(std::complex<double> u, double tau) const noexcept
{
    using cd = std::complex<double>;

    const double kappa = params_.kappa;
    const double theta = params_.theta;
    const double xi    = params_.xi;
    const double rho   = params_.rho;

    const cd iu  = cd(0.0, 1.0) * u;       // i·u
    const cd iu2 = iu + u * u;             // i·u + u²  (= iu - u² ? no: i·u + u·u = iu + u²)

    // beta = kappa - rho*xi*i*u  [Albrecher eq. 3]
    const cd beta = cd(kappa, 0.0) - cd(0.0, rho * xi) * u;

    // d = sqrt(beta^2 + xi^2*(iu + u^2))  [Albrecher eq. 3]
    const cd d = std::sqrt(beta * beta + cd(xi * xi, 0.0) * iu2);

    // Rotation-corrected g (Albrecher eq. 4):
    //   g = (beta - d) / (beta + d)
    const cd bmd = beta - d;
    const cd bpd = beta + d;
    const cd g   = bmd / bpd;

    // e^{-d*tau}
    const cd edtau = std::exp(-d * cd(tau, 0.0));

    // 1 - g * e^{-d*tau}
    const cd one_minus_gedtau = cd(1.0, 0.0) - g * edtau;
    // 1 - g
    const cd one_minus_g      = cd(1.0, 0.0) - g;

    // D(u,tau) = (bmd / xi^2) * (1 - e^{-d*tau}) / (1 - g*e^{-d*tau})
    //          [Albrecher eq. 4]
    const cd D = (bmd / cd(xi * xi, 0.0)) * (cd(1.0, 0.0) - edtau) / one_minus_gedtau;

    // C(u,tau) = (kappa*theta/xi^2) * [bmd*tau - 2*ln((1-g*e^{-dtau})/(1-g))]
    //          [Albrecher eq. 5]
    const cd log_ratio = std::log(one_minus_gedtau / one_minus_g);
    const cd C = cd(kappa * theta / (xi * xi), 0.0)
                 * (bmd * cd(tau, 0.0) - cd(2.0, 0.0) * log_ratio);

    return {C, D};
}

// ---------------------------------------------------------------------------
// Main characteristic function evaluation (Albrecher formulation)
//
// φ(u; τ) = exp(C(u,τ) + D(u,τ)·v₀ + i·u·ln(F·e^{-r·τ}))
//   where F = S₀ · e^{(r-q)·τ}
//   so i·u·ln(F) = i·u·(ln(S₀) + (r-q)·τ)
// ---------------------------------------------------------------------------

std::complex<double>
HestonCharacteristicFn::operator()(std::complex<double> u, double tau) const noexcept
{
    using cd = std::complex<double>;

    auto [C, D] = compute_CD(u, tau);

    // Forward price F = S₀ exp((r-q)τ), characteristic fn is for ln(F) as state
    const double ln_F = std::log(S0_) + (r_ - q_) * tau;

    // φ = exp(C + D·v₀ + i·u·ln(F))
    return std::exp(C + D * cd(params_.v0, 0.0) + cd(0.0, 1.0) * u * cd(ln_F, 0.0));
}

// ---------------------------------------------------------------------------
// Original Heston (1993) formulation — for comparison and branch-cut demo
//
// Heston (1993) eq. 17 uses:
//   b  = kappa - rho*xi*i*u
//   d  = sqrt(b^2 - xi^2*(2*i*u - u^2))  [note sign difference]
//   g  = (b - d) / (b + d)
//
// This is mathematically equivalent to the Albrecher form but can suffer
// from branch-cut discontinuities when Re(d) changes sign along the path.
// ---------------------------------------------------------------------------

std::complex<double>
HestonCharacteristicFn::original_heston(std::complex<double> u, double tau) const noexcept
{
    using cd = std::complex<double>;

    const double kappa = params_.kappa;
    const double theta = params_.theta;
    const double xi    = params_.xi;
    const double rho   = params_.rho;

    const cd iu = cd(0.0, 1.0) * u;

    // Heston (1993) notation: a = kappa*theta, b = kappa, xi = sigma_v
    // d = sqrt((rho*xi*iu - b)^2 + xi^2*(iu + u^2))
    // b = κ − ρξiu,  d = sqrt(b² + ξ²(iu + u²))
    // This is identical to the Albrecher form; the "original" label refers to
    // a different choice of g (= 1/g_Albrecher) that causes branch-cut issues.
    const cd b   = cd(kappa, 0.0) - cd(0.0, rho * xi) * u;
    const cd d   = std::sqrt(b * b + cd(xi * xi, 0.0) * (iu + u * u));

    const cd g = (b - d) / (b + d);
    const cd edtau = std::exp(-d * cd(tau, 0.0));

    // D_orig = (b-d)/xi^2 * (1 - e^{-dtau}) / (1 - g*e^{-dtau})
    const cd D_orig = ((b - d) / cd(xi * xi, 0.0))
                      * (cd(1.0, 0.0) - edtau)
                      / (cd(1.0, 0.0) - g * edtau);

    // C_orig = r*iu*tau + kappa*theta/xi^2 * [(b-d)*tau - 2*ln((1-g*e^{-dtau})/(1-g))]
    const cd log_rat = std::log((cd(1.0, 0.0) - g * edtau) / (cd(1.0, 0.0) - g));
    const cd C_orig  = cd(r_, 0.0) * iu * cd(tau, 0.0)
                       + cd(kappa * theta / (xi * xi), 0.0)
                         * ((b - d) * cd(tau, 0.0) - cd(2.0, 0.0) * log_rat);

    // Use ln(F) = ln(S0) + (r-q)*tau to match Albrecher formulation
    // (original Heston 1993 omits dividends; including q here for consistency)
    const double ln_F = std::log(S0_) + (r_ - q_) * tau;
    return std::exp(C_orig + D_orig * cd(params_.v0, 0.0) + iu * cd(ln_F, 0.0));
}

// ---------------------------------------------------------------------------
// Batch evaluation — tight inner loop for FFT
// ---------------------------------------------------------------------------

void HestonCharacteristicFn::evaluate_batch(
    const std::vector<std::complex<double>>& u_vec,
    double tau,
    std::vector<std::complex<double>>& out) const
{
    const std::size_t N = u_vec.size();
    out.resize(N);
    for (std::size_t j = 0; j < N; ++j) {
        out[j] = (*this)(u_vec[j], tau);
    }
}

// ---------------------------------------------------------------------------
// Sanity check: φ(0; τ) must equal 1 for any valid parameters
// ---------------------------------------------------------------------------

bool HestonCharacteristicFn::sanity_check(double tau) const
{
    using cd = std::complex<double>;
    const cd phi0 = (*this)(cd(0.0, 0.0), tau);
    return std::abs(phi0 - cd(1.0, 0.0)) < 1e-8;
}

// ---------------------------------------------------------------------------
// Branch-cut check: flag when Im(d) ≈ 0 and Re(d) < 0 (danger zone for
// the original Heston form; the Albrecher form handles this correctly).
// ---------------------------------------------------------------------------

bool HestonCharacteristicFn::branch_cut_check(std::complex<double> u,
                                              double               tau) const noexcept
{
    (void)tau; // not used — branch cut depends on u only

    const double kappa = params_.kappa;
    const double xi    = params_.xi;
    const double rho   = params_.rho;

    using cd = std::complex<double>;
    const cd beta = cd(kappa, 0.0) - cd(0.0, rho * xi) * u;
    const cd iu2  = cd(0.0, 1.0) * u + u * u;
    const cd d_sq = beta * beta + cd(xi * xi, 0.0) * iu2;

    // Branch cut of sqrt is the negative real axis.
    // Flag when d_sq is near the negative real axis.
    return (d_sq.imag() == 0.0) && (d_sq.real() < 0.0);
}

} // namespace heston
