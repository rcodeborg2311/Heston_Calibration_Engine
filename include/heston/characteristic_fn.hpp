#pragma once
/**
 * @file characteristic_fn.hpp
 * @brief Heston (1993) characteristic function φ(u; τ).
 *
 * Implements the numerically stable "little Heston trap" formulation from
 * Albrecher, Mayer, Schachermayer & Teugels (2007) to avoid the branch-cut
 * discontinuity that arises in the original Heston (1993) form when the
 * integration contour crosses a pole.
 *
 * References:
 *   [1] Heston, S. L. (1993). "A Closed-Form Solution for Options with
 *       Stochastic Volatility," Rev. Financial Studies 6(2), 327-343.
 *   [2] Albrecher, H., Mayer, P., Schachermayer, W. & Teugels, J. (2007).
 *       "The little Heston trap," Wilmott Magazine, Jan 2007, 83-92.
 */

#include "types.hpp"
#include <complex>
#include <utility>
#include <vector>

namespace heston {

/**
 * @brief Evaluates the Heston risk-neutral characteristic function.
 *
 * φ(u; τ, S, v₀) = exp(C(u,τ) + D(u,τ)·v₀ + i·u·ln(S·e^{rτ}))
 *
 * where C and D are computed via the rotation-corrected ("little Heston trap")
 * formulation of Albrecher et al. (2007) [Ref 2, eq. 4-5].
 *
 * This class is designed to be called millions of times per calibration run.
 * All hot-path methods are inline-friendly and allocation-free.
 */
class HestonCharacteristicFn {
public:
    /**
     * @brief Construct from model parameters and market data.
     * @param p   Heston model parameters {v0, kappa, theta, xi, rho}
     * @param mkt Market data supplying S0, r, q
     */
    HestonCharacteristicFn(const HestonParams& p, const MarketData& mkt);

    /**
     * @brief Evaluate φ(u; τ) at a single complex argument.
     *
     * Implements Albrecher et al. (2007) eq. 4-5 (rotation-corrected form).
     * This is the primary hot-path function.
     *
     * @param u   Complex integration variable
     * @param tau Time to expiry τ = T - t  (years, must be > 0)
     * @return    Complex characteristic function value
     */
    [[nodiscard]] std::complex<double>
    operator()(std::complex<double> u, double tau) const noexcept;

    /**
     * @brief Batch evaluation for FFT: fills out[j] = φ(u_j; tau).
     *
     * Avoids repeated function-call overhead for the FFT inner loop.
     * @param u_vec Input complex frequencies
     * @param tau   Common time to expiry
     * @param out   Output vector (resized to match u_vec if needed)
     */
    void evaluate_batch(const std::vector<std::complex<double>>& u_vec,
                        double tau,
                        std::vector<std::complex<double>>& out) const;

    /**
     * @brief Evaluate using the original Heston (1993) formulation.
     *
     * The original form uses g = (b-d)/(b+d) and can have branch-cut issues
     * for long maturities or extreme parameters. Retained for comparison.
     *
     * Reference: Heston (1993) [Ref 1], eq. 17.
     *
     * @param u   Complex integration variable
     * @param tau Time to expiry
     * @return    Characteristic function value (original formulation)
     */
    [[nodiscard]] std::complex<double>
    original_heston(std::complex<double> u, double tau) const noexcept;

    /**
     * @brief Check that φ(0; τ) = 1 (martingale condition).
     *
     * This must hold for any valid parameter set and serves as a basic
     * sanity check on the implementation.
     *
     * @param tau Time to expiry (any positive value)
     * @return    true if |φ(0;τ) - 1| < 1e-10
     */
    [[nodiscard]] bool sanity_check(double tau) const;

    /**
     * @brief Detect and flag potential branch-cut crossings.
     *
     * Returns true if the argument d(u) computed along the integration
     * path at the given u is likely to cross the negative real axis,
     * indicating the original Heston formulation would give wrong results.
     *
     * @param u   Complex integration variable to test
     * @param tau Time to expiry
     * @return    true if branch-cut correction is needed
     */
    [[nodiscard]] bool branch_cut_check(std::complex<double> u, double tau) const noexcept;

    // Accessors for downstream use (e.g., Greeks pathwise differentiation)
    [[nodiscard]] const HestonParams& params() const noexcept { return params_; }
    [[nodiscard]] double r()  const noexcept { return r_; }
    [[nodiscard]] double q()  const noexcept { return q_; }
    [[nodiscard]] double S0() const noexcept { return S0_; }

    /**
     * @brief Compute (C, D) coefficients of the log-characteristic function.
     *
     * φ(u;τ) = exp(C(u,τ) + D(u,τ)·v₀ + i·u·ln(F))
     *
     * Uses the Albrecher et al. rotation-corrected form throughout.
     * D is returned directly for use in the vega pathwise formula:
     *   ∂φ/∂v₀ = D(u,τ) · φ(u,τ)
     *
     * Reference: Albrecher et al. (2007) [Ref 2], eqs. 4-5.
     *
     * @param u   Complex integration variable
     * @param tau Time to expiry
     * @return    Pair {C(u,τ), D(u,τ)}
     */
    [[nodiscard]] std::pair<std::complex<double>, std::complex<double>>
    compute_CD(std::complex<double> u, double tau) const noexcept;

private:
    HestonParams params_;
    double r_, q_, S0_;
};

} // namespace heston
