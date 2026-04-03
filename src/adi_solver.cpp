/**
 * @file adi_solver.cpp
 * @brief Craig-Sneyd ADI finite-difference solver implementation.
 *
 * Reference:
 *   In 't Hout, K. J. & Foulon, S. (2010). "ADI finite difference schemes for
 *   option pricing in the Heston model with correlation," Int. J. Numer.
 *   Anal. Model. 7(2), 303-320.
 *
 * The Craig-Sneyd (CS) scheme with θ=0.5 achieves second-order accuracy in
 * both space and time.  It handles the mixed derivative ∂²u/∂S∂v explicitly,
 * which is valid for ρ ∈ (−1, 1) and does not require additional implicit
 * treatment of the cross term.
 *
 * Grid coordinates:
 *   i ∈ {0,...,Ns-1}  →  S_i (non-uniform, concentrated near S₀)
 *   j ∈ {0,...,Nv-1}  →  v_j (non-uniform, concentrated near v₀)
 *   U[i + j*Ns] = option value at (S_i, v_j)
 *
 * The PDE in backward time τ = T − t:
 *   ∂u/∂τ = ½vS²u_SS + ρξvS u_Sv + ½ξ²v u_vv + (r-q)S u_S + κ(θ-v) u_v - ru
 *
 * Operator splitting (Craig-Sneyd):
 *   L₁ = S-direction: ½vS²∂²/∂S² + (r-q)S∂/∂S − ½r I
 *   L₂ = v-direction: ½ξ²v∂²/∂v² + κ(θ-v)∂/∂v − ½r I
 *   F₀ = mixed term: ρξvS ∂²/∂S∂v  (treated explicitly)
 */

#include "heston/adi_solver.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace heston {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

ADISolver::ADISolver(const HestonParams& params, const MarketData& mkt)
    : ADISolver(params, mkt, Config{})
{}

ADISolver::ADISolver(const HestonParams& params, const MarketData& mkt, Config cfg)
    : params_(params), mkt_(mkt), cfg_(cfg)
{
    build_grids();
}

// ---------------------------------------------------------------------------
// Grid construction
//
// S-grid: non-uniform, sinh spacing concentrated near S₀
//   S_i = S₀ · sinh(c_s · i / (Ns-1)) / sinh(c_s)   for i=0,...,Ns-1
//   i=0 maps to S=0, i=Ns-1 maps to S ≈ S₀·1 = S₀ * sinhmax/sinhmax... hmm
//
// Actually the standard formula for a Heston S-grid is:
//   S_i = Smax · sinh(c_s · i / Ns) / sinh(c_s)
// That maps i=0→0, i=Ns→Smax.  We use Ns+1 internal + boundary points.
//
// v-grid: similar with v₀-concentration
//   v_j = vmax · sinh(c_v · j / Nv) / sinh(c_v)
// ---------------------------------------------------------------------------

void ADISolver::build_grids()
{
    const int    Ns    = cfg_.Ns;
    const int    Nv    = cfg_.Nv;
    const double Smax  = cfg_.Smax_mult * mkt_.spot;
    const double vmax  = cfg_.vmax_mult * params_.theta;
    const double c_s   = cfg_.c_s;
    const double c_v   = cfg_.c_v;
    const double sinh_cs = std::sinh(c_s);
    const double sinh_cv = std::sinh(c_v);

    S_grid_.resize(static_cast<std::size_t>(Ns));
    v_grid_.resize(static_cast<std::size_t>(Nv));

    for (int i = 0; i < Ns; ++i) {
        // S_i = Smax * sinh(c_s * i / (Ns-1)) / sinh(c_s)
        const double xi = static_cast<double>(i) / static_cast<double>(Ns - 1);
        S_grid_[static_cast<std::size_t>(i)] = Smax * std::sinh(c_s * xi) / sinh_cs;
    }

    for (int j = 0; j < Nv; ++j) {
        const double xj = static_cast<double>(j) / static_cast<double>(Nv - 1);
        v_grid_[static_cast<std::size_t>(j)] = vmax * std::sinh(c_v * xj) / sinh_cv;
    }
}

// ---------------------------------------------------------------------------
// Thomas algorithm (tridiagonal solve)
//
// a[i] · x[i-1] + b[i] · x[i] + c[i] · x[i+1] = d[i]
//
// Forward sweep: eliminate subdiagonal
// Back substitution: recover x from d (modified in-place)
// ---------------------------------------------------------------------------

void ADISolver::thomas_solve(std::vector<double>& a,
                             std::vector<double>& b,
                             std::vector<double>& c,
                             std::vector<double>& d) noexcept
{
    const int n = static_cast<int>(d.size());
    if (n == 0) return;

    // Forward elimination
    for (int k = 1; k < n; ++k) {
        const double m = a[static_cast<std::size_t>(k)] / b[static_cast<std::size_t>(k - 1)];
        b[static_cast<std::size_t>(k)] -= m * c[static_cast<std::size_t>(k - 1)];
        d[static_cast<std::size_t>(k)] -= m * d[static_cast<std::size_t>(k - 1)];
    }

    // Back substitution
    d[static_cast<std::size_t>(n - 1)] /= b[static_cast<std::size_t>(n - 1)];
    for (int k = n - 2; k >= 0; --k) {
        d[static_cast<std::size_t>(k)] =
            (d[static_cast<std::size_t>(k)]
             - c[static_cast<std::size_t>(k)] * d[static_cast<std::size_t>(k + 1)])
            / b[static_cast<std::size_t>(k)];
    }
}

// ---------------------------------------------------------------------------
// Terminal condition: u(T, S, v) = max(S − K, 0)
// ---------------------------------------------------------------------------

void ADISolver::apply_terminal_condition(double K, std::vector<double>& U) const
{
    const int Ns = cfg_.Ns;
    const int Nv = cfg_.Nv;

    for (int j = 0; j < Nv; ++j) {
        for (int i = 0; i < Ns; ++i) {
            U[static_cast<std::size_t>(i + j * Ns)] =
                std::max(S_grid_[static_cast<std::size_t>(i)] - K, 0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Boundary conditions
//
//   S = 0   (i=0):         u = 0  (call worthless)
//   S = Smax (i=Ns-1):     u = Smax*e^{-qτ} − K*e^{-rτ}
//   v = 0   (j=0):         ∂u/∂v = 0  (degenerate PDE reduces to BS with σ→0;
//                          enforce by keeping the 1D implicit solve)
//   v = vmax (j=Nv-1):     ∂u/∂v = 0  (vanishing vol effect at high variance)
// ---------------------------------------------------------------------------

void ADISolver::apply_boundary_conditions(std::vector<double>& U,
                                          double K, double r, double q,
                                          double tau) const
{
    const int    Ns   = cfg_.Ns;
    const int    Nv   = cfg_.Nv;
    const double Smax = cfg_.Smax_mult * mkt_.spot;

    const double bc_Smax = Smax * std::exp(-q * tau) - K * std::exp(-r * tau);
    const double bc_S0   = 0.0;

    for (int j = 0; j < Nv; ++j) {
        // S = 0 boundary
        U[static_cast<std::size_t>(0 + j * Ns)] = bc_S0;
        // S = Smax boundary
        U[static_cast<std::size_t>((Ns - 1) + j * Ns)] = std::max(bc_Smax, 0.0);
    }

    // v = 0: zero-flux (Neumann) — enforce by copying from j=1
    for (int i = 0; i < Ns; ++i) {
        U[static_cast<std::size_t>(i + 0 * Ns)] = U[static_cast<std::size_t>(i + 1 * Ns)];
    }

    // v = vmax: zero-flux — enforce by copying from j=Nv-2
    for (int i = 0; i < Ns; ++i) {
        U[static_cast<std::size_t>(i + (Nv - 1) * Ns)] =
            U[static_cast<std::size_t>(i + (Nv - 2) * Ns)];
    }
}

// ---------------------------------------------------------------------------
// Bilinear interpolation to (S₀, v₀)
// ---------------------------------------------------------------------------

double ADISolver::interpolate_to_spot(const std::vector<double>& U) const
{
    const int    Ns = cfg_.Ns;
    const double S0 = mkt_.spot;
    const double v0 = params_.v0;

    // Find surrounding S-indices
    const auto is_it = std::lower_bound(S_grid_.begin(), S_grid_.end(), S0);
    std::size_t is = static_cast<std::size_t>(is_it - S_grid_.begin());
    if (is == 0) is = 1;
    if (is >= static_cast<std::size_t>(Ns)) is = static_cast<std::size_t>(Ns - 1);
    const std::size_t is0 = is - 1;

    // Find surrounding v-indices
    const auto jv_it = std::lower_bound(v_grid_.begin(), v_grid_.end(), v0);
    std::size_t jv = static_cast<std::size_t>(jv_it - v_grid_.begin());
    if (jv == 0) jv = 1;
    if (jv >= static_cast<std::size_t>(cfg_.Nv)) jv = static_cast<std::size_t>(cfg_.Nv - 1);
    const std::size_t jv0 = jv - 1;

    // Bilinear weights
    const double dS = S_grid_[is] - S_grid_[is0];
    const double dv = v_grid_[jv] - v_grid_[jv0];
    const double wS = (dS > 0.0) ? (S0 - S_grid_[is0]) / dS : 0.5;
    const double wv = (dv > 0.0) ? (v0 - v_grid_[jv0]) / dv : 0.5;

    const double u00 = U[is0 + jv0 * static_cast<std::size_t>(Ns)];
    const double u10 = U[is  + jv0 * static_cast<std::size_t>(Ns)];
    const double u01 = U[is0 + jv  * static_cast<std::size_t>(Ns)];
    const double u11 = U[is  + jv  * static_cast<std::size_t>(Ns)];

    return (1.0 - wS) * (1.0 - wv) * u00
         + wS         * (1.0 - wv) * u10
         + (1.0 - wS) * wv         * u01
         + wS         * wv         * u11;
}

// ---------------------------------------------------------------------------
// Core Craig-Sneyd ADI solve
//
// Six-step algorithm (In 't Hout & Foulon 2010, §4):
//   Y₀ = Uⁿ + Δτ·(L₁+L₂+F₀)·Uⁿ                   [explicit full step]
//   (I − θ·Δτ·L₁)·Y₁ = Y₀ − θ·Δτ·L₁·Uⁿ            [implicit S-direction]
//   (I − θ·Δτ·L₂)·Y₂ = Y₁ − θ·Δτ·L₂·Uⁿ            [implicit v-direction]
//   Ỹ₀ = Y₀ + θ·Δτ·(L₁+L₂)·(Y₂−Uⁿ)                [correction]
//   (I − θ·Δτ·L₁)·Ỹ₁ = Ỹ₀ − θ·Δτ·L₁·Y₂            [second S-implicit]
//   Uⁿ⁺¹ = Ỹ₁
// ---------------------------------------------------------------------------

double ADISolver::solve(double K, double T, int Nt) const
{
    const int    Ns      = cfg_.Ns;
    const int    Nv      = cfg_.Nv;
    const double theta_p = params_.theta;
    const double kappa   = params_.kappa;
    const double xi      = params_.xi;
    const double rho     = params_.rho;
    const double r       = mkt_.rate;
    const double q       = mkt_.div_yield;
    const double theta_cs = cfg_.theta_cs; // CS blending (0.5)

    const double dtau = T / static_cast<double>(Nt);
    const std::size_t sz = static_cast<std::size_t>(Ns * Nv);

    // Solution array
    std::vector<double> U(sz), Y0(sz), Y1(sz), Y2(sz), Y0t(sz), Y1t(sz);
    std::vector<double> L1U(sz, 0.0), L2U(sz, 0.0), F0U(sz, 0.0);

    // Tridiagonal workspace (reused each sweep)
    std::vector<double> a(static_cast<std::size_t>(std::max(Ns, Nv)));
    std::vector<double> b(static_cast<std::size_t>(std::max(Ns, Nv)));
    std::vector<double> c(static_cast<std::size_t>(std::max(Ns, Nv)));
    std::vector<double> d(static_cast<std::size_t>(std::max(Ns, Nv)));

    // Initialize terminal condition
    apply_terminal_condition(K, U);

    // Helper: second-order finite difference coefficients for non-uniform grids
    // Non-uniform second derivative (Fornberg 1988):
    //   ∂²u/∂x² ≈  2·u_{i-1}/(hm(hm+hp))
    //             − 2·u_i/(hm·hp)
    //             + 2·u_{i+1}/(hp(hm+hp))
    auto diff2_coeff = [](double hm, double hp) -> std::array<double, 3> {
        return { 2.0 / (hm * (hm + hp)),
                -2.0 / (hm * hp),
                 2.0 / (hp * (hm + hp)) };
    };

    auto diff1_coeff = [](double hm, double hp) -> std::array<double, 3> {
        // Central difference for ∂u/∂x: a·u_{i-1} + 0·u_i + c·u_{i+1}
        return { -hp / (hm * (hm + hp)),
                  (hp - hm) / (hm * hp),
                  hm / (hp * (hm + hp)) };
    };

    // -----------------------------------------------------------------------
    // Time-stepping loop (backward from τ=0 to τ=T)
    // -----------------------------------------------------------------------

    for (int nt = 0; nt < Nt; ++nt) {
        const double tau = dtau * static_cast<double>(nt + 1); // time elapsed from maturity

        // ---- Compute L₁U, L₂U, F₀U at current U ----

        // Interior points only (boundary handled separately)
        for (int j = 1; j < Nv - 1; ++j) {
            const double vj  = v_grid_[static_cast<std::size_t>(j)];
            const double hvm = vj - v_grid_[static_cast<std::size_t>(j - 1)];
            const double hvp = v_grid_[static_cast<std::size_t>(j + 1)] - vj;

            for (int i = 1; i < Ns - 1; ++i) {
                const double Si  = S_grid_[static_cast<std::size_t>(i)];
                const double hSm = Si - S_grid_[static_cast<std::size_t>(i - 1)];
                const double hSp = S_grid_[static_cast<std::size_t>(i + 1)] - Si;

                const std::size_t idx = static_cast<std::size_t>(i + j * Ns);
                const std::size_t im1 = static_cast<std::size_t>((i-1) + j * Ns);
                const std::size_t ip1 = static_cast<std::size_t>((i+1) + j * Ns);
                const std::size_t jm1 = static_cast<std::size_t>(i + (j-1) * Ns);
                const std::size_t jp1 = static_cast<std::size_t>(i + (j+1) * Ns);

                // Second derivatives (non-uniform)
                auto [aS2, bS2, cS2] = diff2_coeff(hSm, hSp);
                auto [av2, bv2, cv2] = diff2_coeff(hvm, hvp);
                // First derivatives (non-uniform)
                auto [aS1, bS1, cS1] = diff1_coeff(hSm, hSp);
                auto [av1, bv1, cv1] = diff1_coeff(hvm, hvp);

                // L₁ = ½·v·S²·∂²/∂S² + (r-q)·S·∂/∂S − ½r
                const double coS2 = 0.5 * vj * Si * Si;
                const double coS1 = (r - q) * Si;
                L1U[idx] = coS2 * (aS2 * U[im1] + bS2 * U[idx] + cS2 * U[ip1])
                         + coS1 * (aS1 * U[im1] + bS1 * U[idx] + cS1 * U[ip1])
                         - 0.5 * r * U[idx];

                // L₂ = ½·ξ²·v·∂²/∂v² + κ(θ−v)·∂/∂v − ½r
                const double cov2 = 0.5 * xi * xi * vj;
                const double cov1 = kappa * (theta_p - vj);
                L2U[idx] = cov2 * (av2 * U[jm1] + bv2 * U[idx] + cv2 * U[jp1])
                         + cov1 * (av1 * U[jm1] + bv1 * U[idx] + cv1 * U[jp1])
                         - 0.5 * r * U[idx];

                // Mixed term F₀ = ρ·ξ·v·S·∂²/∂S∂v  (explicit)
                // ∂²u/∂S∂v ≈ (u_{i+1,j+1} - u_{i+1,j-1} - u_{i-1,j+1} + u_{i-1,j-1}) / (4 hS hv)
                const std::size_t ip1jp1 = static_cast<std::size_t>((i+1) + (j+1) * Ns);
                const std::size_t ip1jm1 = static_cast<std::size_t>((i+1) + (j-1) * Ns);
                const std::size_t im1jp1 = static_cast<std::size_t>((i-1) + (j+1) * Ns);
                const std::size_t im1jm1 = static_cast<std::size_t>((i-1) + (j-1) * Ns);

                const double d2uSv = (U[ip1jp1] - U[ip1jm1] - U[im1jp1] + U[im1jm1])
                                     / (4.0 * ((hSm + hSp) * 0.5) * ((hvm + hvp) * 0.5));

                F0U[idx] = rho * xi * vj * Si * d2uSv;
            }
        }

        // ---- Step 1: Y₀ = U + Δτ·(L₁ + L₂ + F₀)·U ----
        for (std::size_t k = 0; k < sz; ++k) {
            Y0[k] = U[k] + dtau * (L1U[k] + L2U[k] + F0U[k]);
        }

        // ---- Step 2: (I − θΔτ·L₁)·Y₁ = Y₀ − θΔτ·L₁·U ----
        // Implicit in S-direction: for each v-row j, solve tridiagonal in S
        {
            // RHS: rhs[i] = Y0[i,j] - theta_cs*dtau*L1U[i,j]
            for (int j = 1; j < Nv - 1; ++j) {
                // Set up RHS
                d.resize(static_cast<std::size_t>(Ns));
                for (int i = 0; i < Ns; ++i) {
                    const std::size_t idx = static_cast<std::size_t>(i + j * Ns);
                    d[static_cast<std::size_t>(i)] = Y0[idx] - theta_cs * dtau * L1U[idx];
                }

                // Build tridiagonal for (I - theta_cs*dtau*L1) in S
                a.assign(static_cast<std::size_t>(Ns), 0.0);
                b.assign(static_cast<std::size_t>(Ns), 1.0);
                c.assign(static_cast<std::size_t>(Ns), 0.0);

                const double vj = v_grid_[static_cast<std::size_t>(j)];

                for (int i = 1; i < Ns - 1; ++i) {
                    const double Si  = S_grid_[static_cast<std::size_t>(i)];
                    const double hSm = Si - S_grid_[static_cast<std::size_t>(i - 1)];
                    const double hSp = S_grid_[static_cast<std::size_t>(i + 1)] - Si;
                    auto [aS2, bS2, cS2] = diff2_coeff(hSm, hSp);
                    auto [aS1, bS1, cS1] = diff1_coeff(hSm, hSp);

                    const double coS2 = 0.5 * vj * Si * Si;
                    const double coS1 = (r - q) * Si;

                    const double lo  = coS2 * aS2 + coS1 * aS1;
                    const double mid = coS2 * bS2 + coS1 * bS1 - 0.5 * r;
                    const double hi  = coS2 * cS2 + coS1 * cS1;

                    a[static_cast<std::size_t>(i)] = -theta_cs * dtau * lo;
                    b[static_cast<std::size_t>(i)] = 1.0 - theta_cs * dtau * mid;
                    c[static_cast<std::size_t>(i)] = -theta_cs * dtau * hi;
                }

                thomas_solve(a, b, c, d);

                for (int i = 0; i < Ns; ++i) {
                    Y1[static_cast<std::size_t>(i + j * Ns)] = d[static_cast<std::size_t>(i)];
                }
            }
            // Copy boundary rows
            for (int i = 0; i < Ns; ++i) {
                Y1[static_cast<std::size_t>(i + 0 * Ns)] = Y0[static_cast<std::size_t>(i)];
                Y1[static_cast<std::size_t>(i + (Nv-1) * Ns)] = Y0[static_cast<std::size_t>(i + (Nv-1) * Ns)];
            }
        }

        // ---- Step 3: (I − θΔτ·L₂)·Y₂ = Y₁ − θΔτ·L₂·U ----
        // Implicit in v-direction: for each S-column i, solve tridiagonal in v
        {
            for (int i = 1; i < Ns - 1; ++i) {
                d.resize(static_cast<std::size_t>(Nv));
                for (int j = 0; j < Nv; ++j) {
                    const std::size_t idx = static_cast<std::size_t>(i + j * Ns);
                    d[static_cast<std::size_t>(j)] = Y1[idx] - theta_cs * dtau * L2U[idx];
                }

                a.assign(static_cast<std::size_t>(Nv), 0.0);
                b.assign(static_cast<std::size_t>(Nv), 1.0);
                c.assign(static_cast<std::size_t>(Nv), 0.0);

                for (int j = 1; j < Nv - 1; ++j) {
                    const double vj  = v_grid_[static_cast<std::size_t>(j)];
                    const double hvm = vj - v_grid_[static_cast<std::size_t>(j - 1)];
                    const double hvp = v_grid_[static_cast<std::size_t>(j + 1)] - vj;
                    auto [av2, bv2, cv2] = diff2_coeff(hvm, hvp);
                    auto [av1, bv1, cv1] = diff1_coeff(hvm, hvp);

                    const double cov2 = 0.5 * xi * xi * vj;
                    const double cov1 = kappa * (theta_p - vj);

                    const double lo  = cov2 * av2 + cov1 * av1;
                    const double mid = cov2 * bv2 + cov1 * bv1 - 0.5 * r;
                    const double hi  = cov2 * cv2 + cov1 * cv1;

                    a[static_cast<std::size_t>(j)] = -theta_cs * dtau * lo;
                    b[static_cast<std::size_t>(j)] = 1.0 - theta_cs * dtau * mid;
                    c[static_cast<std::size_t>(j)] = -theta_cs * dtau * hi;
                }

                thomas_solve(a, b, c, d);

                for (int j = 0; j < Nv; ++j) {
                    Y2[static_cast<std::size_t>(i + j * Ns)] = d[static_cast<std::size_t>(j)];
                }
            }
            for (int j = 0; j < Nv; ++j) {
                Y2[static_cast<std::size_t>(0 + j * Ns)] = Y1[static_cast<std::size_t>(j * Ns)];
                Y2[static_cast<std::size_t>((Ns-1) + j * Ns)] = Y1[static_cast<std::size_t>((Ns-1) + j * Ns)];
            }
        }

        // ---- Step 4: Ỹ₀ = Y₀ + θΔτ(L₁+L₂)(Y₂ − U) ----
        // Re-compute L₁ and L₂ applied to (Y₂ − U)
        {
            std::vector<double> diff(sz);
            for (std::size_t k = 0; k < sz; ++k) diff[k] = Y2[k] - U[k];

            // Reset operators
            std::fill(L1U.begin(), L1U.end(), 0.0);
            std::fill(L2U.begin(), L2U.end(), 0.0);

            for (int j = 1; j < Nv - 1; ++j) {
                const double vj = v_grid_[static_cast<std::size_t>(j)];
                const double hvm = vj - v_grid_[static_cast<std::size_t>(j - 1)];
                const double hvp = v_grid_[static_cast<std::size_t>(j + 1)] - vj;
                auto [av2, bv2, cv2] = diff2_coeff(hvm, hvp);
                auto [av1, bv1, cv1] = diff1_coeff(hvm, hvp);

                for (int i = 1; i < Ns - 1; ++i) {
                    const double Si  = S_grid_[static_cast<std::size_t>(i)];
                    const double hSm = Si - S_grid_[static_cast<std::size_t>(i - 1)];
                    const double hSp = S_grid_[static_cast<std::size_t>(i + 1)] - Si;
                    auto [aS2, bS2, cS2] = diff2_coeff(hSm, hSp);
                    auto [aS1, bS1, cS1] = diff1_coeff(hSm, hSp);

                    const std::size_t idx = static_cast<std::size_t>(i + j * Ns);
                    const std::size_t im1 = static_cast<std::size_t>((i-1) + j * Ns);
                    const std::size_t ip1 = static_cast<std::size_t>((i+1) + j * Ns);
                    const std::size_t jm1 = static_cast<std::size_t>(i + (j-1) * Ns);
                    const std::size_t jp1 = static_cast<std::size_t>(i + (j+1) * Ns);

                    L1U[idx] = (0.5 * vj * Si * Si) * (aS2 * diff[im1] + bS2 * diff[idx] + cS2 * diff[ip1])
                             + ((r - q) * Si) * (aS1 * diff[im1] + bS1 * diff[idx] + cS1 * diff[ip1])
                             - 0.5 * r * diff[idx];

                    L2U[idx] = (0.5 * xi * xi * vj) * (av2 * diff[jm1] + bv2 * diff[idx] + cv2 * diff[jp1])
                             + (kappa * (theta_p - vj)) * (av1 * diff[jm1] + bv1 * diff[idx] + cv1 * diff[jp1])
                             - 0.5 * r * diff[idx];
                }
            }

            for (std::size_t k = 0; k < sz; ++k) {
                Y0t[k] = Y0[k] + theta_cs * dtau * (L1U[k] + L2U[k]);
            }
        }

        // ---- Step 5: (I − θΔτ·L₁)·Ỹ₁ = Ỹ₀ − θΔτ·L₁·Y₂ ----
        // Re-compute L₁·Y₂
        {
            std::fill(L1U.begin(), L1U.end(), 0.0);
            for (int j = 1; j < Nv - 1; ++j) {
                const double vj = v_grid_[static_cast<std::size_t>(j)];
                for (int i = 1; i < Ns - 1; ++i) {
                    const double Si  = S_grid_[static_cast<std::size_t>(i)];
                    const double hSm = Si - S_grid_[static_cast<std::size_t>(i - 1)];
                    const double hSp = S_grid_[static_cast<std::size_t>(i + 1)] - Si;
                    auto [aS2, bS2, cS2] = diff2_coeff(hSm, hSp);
                    auto [aS1, bS1, cS1] = diff1_coeff(hSm, hSp);

                    const std::size_t idx = static_cast<std::size_t>(i + j * Ns);
                    const std::size_t im1 = static_cast<std::size_t>((i-1) + j * Ns);
                    const std::size_t ip1 = static_cast<std::size_t>((i+1) + j * Ns);

                    L1U[idx] = (0.5 * vj * Si * Si) * (aS2 * Y2[im1] + bS2 * Y2[idx] + cS2 * Y2[ip1])
                             + ((r - q) * Si) * (aS1 * Y2[im1] + bS1 * Y2[idx] + cS1 * Y2[ip1])
                             - 0.5 * r * Y2[idx];
                }
            }

            // Implicit S-solve with Y0t
            for (int j = 1; j < Nv - 1; ++j) {
                const double vj = v_grid_[static_cast<std::size_t>(j)];
                d.resize(static_cast<std::size_t>(Ns));
                for (int i = 0; i < Ns; ++i) {
                    const std::size_t idx = static_cast<std::size_t>(i + j * Ns);
                    d[static_cast<std::size_t>(i)] = Y0t[idx] - theta_cs * dtau * L1U[idx];
                }

                a.assign(static_cast<std::size_t>(Ns), 0.0);
                b.assign(static_cast<std::size_t>(Ns), 1.0);
                c.assign(static_cast<std::size_t>(Ns), 0.0);

                for (int i = 1; i < Ns - 1; ++i) {
                    const double Si  = S_grid_[static_cast<std::size_t>(i)];
                    const double hSm = Si - S_grid_[static_cast<std::size_t>(i - 1)];
                    const double hSp = S_grid_[static_cast<std::size_t>(i + 1)] - Si;
                    auto [aS2, bS2, cS2] = diff2_coeff(hSm, hSp);
                    auto [aS1, bS1, cS1] = diff1_coeff(hSm, hSp);

                    const double coS2 = 0.5 * vj * Si * Si;
                    const double coS1 = (r - q) * Si;

                    a[static_cast<std::size_t>(i)] = -theta_cs * dtau * (coS2 * aS2 + coS1 * aS1);
                    b[static_cast<std::size_t>(i)] = 1.0 - theta_cs * dtau * (coS2 * bS2 + coS1 * bS1 - 0.5 * r);
                    c[static_cast<std::size_t>(i)] = -theta_cs * dtau * (coS2 * cS2 + coS1 * cS1);
                }

                thomas_solve(a, b, c, d);

                for (int i = 0; i < Ns; ++i) {
                    Y1t[static_cast<std::size_t>(i + j * Ns)] = d[static_cast<std::size_t>(i)];
                }
            }
            for (int i = 0; i < Ns; ++i) {
                Y1t[static_cast<std::size_t>(i)] = Y0t[static_cast<std::size_t>(i)];
                Y1t[static_cast<std::size_t>(i + (Nv-1) * Ns)] = Y0t[static_cast<std::size_t>(i + (Nv-1) * Ns)];
            }
        }

        // ---- Step 6: Uⁿ⁺¹ = Ỹ₁ ----
        U = Y1t;

        // Reapply boundary conditions
        apply_boundary_conditions(U, K, r, q, tau);
    }

    return std::max(interpolate_to_spot(U), 0.0);
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

double ADISolver::call_price(double K, double T) const
{
    return solve(K, T, cfg_.Nt);
}

ADISolver::ConvergenceResult ADISolver::convergence_study(double K, double T) const
{
    ConvergenceResult res;
    res.Nt_values = {50, 100, 200, 400};

    for (int Nt : res.Nt_values) {
        res.prices.push_back(solve(K, T, Nt));
    }

    // Richardson extrapolation using last three values
    // Assuming O(h²): P = P_h + A·h² + ...
    // P(h) ≈ P(h/2) + (P(h/2)-P(h))·4/3 ... (standard Richardson)
    const double p1 = res.prices[2]; // Nt=200
    const double p2 = res.prices[3]; // Nt=400  (finer by 2x)
    res.extrapolated_price = p2 + (p2 - p1) / 3.0; // Richardson with p=2

    // Order of convergence from three consecutive refinements
    const double p0 = res.prices[1]; // Nt=100
    if (std::abs(p1 - p0) > 1e-12 && std::abs(p2 - p1) > 1e-12) {
        res.order_of_convergence = std::log2(std::abs(p1 - p0) / std::abs(p2 - p1));
    } else {
        res.order_of_convergence = 2.0; // asymptotic
    }

    return res;
}

} // namespace heston
