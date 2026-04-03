#pragma once
/**
 * @file greeks.hpp
 * @brief Heston model option Greeks via pathwise differentiation.
 *
 * Greeks are computed analytically by differentiating the Carr-Madan FFT
 * pricing integral with respect to each input.  This is more accurate and
 * faster than finite differences for the primary Greeks (Δ, ν, θ, ρ) and
 * is referred to as "pathwise differentiation" or "infinitesimal perturbation
 * analysis" in the simulation literature.
 *
 * For second-order Greeks (Vanna, Volga) we differentiate the closed-form
 * first-order expressions.
 *
 * Reference: Broadie, M. & Glasserman, P. (1996). "Estimating Security Price
 * Derivatives Using Simulation," Management Science 42(2), 269-285.
 */

#include "characteristic_fn.hpp"
#include "types.hpp"
#include <vector>

namespace heston {

/**
 * @brief Option sensitivities for one (K, T) pair.
 */
struct Greeks {
    double delta; ///< ∂C/∂S   — hedge ratio
    double vega;  ///< ∂C/∂v₀  — sensitivity to initial variance
    double vanna; ///< ∂²C/∂S∂σ — cross sensitivity (skew hedge)
    double volga; ///< ∂²C/∂v₀² — vol-of-vol sensitivity
    double theta; ///< −∂C/∂T  — time decay (positive convention)
    double rho;   ///< ∂C/∂r   — interest-rate sensitivity

    void print() const;
};

/**
 * @brief Compute Heston model Greeks using pathwise differentiation.
 *
 * Pathwise formulae (Δ example):
 *   C(S,v₀) = (e^{-αk}/π) Re[FFT(ψ)]  where k = ln(K/F), F = S·e^{(r-q)T}
 *   ∂C/∂S = (e^{-αk}/π) Re[FFT(∂ψ/∂S)]
 *          + term from dk/dS = -1/S
 *
 * Vega pathwise formula (uses D from characteristic function):
 *   ∂φ/∂v₀ = D(u,τ) · φ(u,τ)
 *   ∂ψ/∂v₀ = e^{-rT} · (∂φ/∂v₀) / denominator(u)
 */
class GreeksCalculator {
public:
    /**
     * @brief Construct from calibrated parameters and market data.
     */
    GreeksCalculator(const HestonParams& params, const MarketData& mkt);

    /**
     * @brief Compute all Greeks for one option using pathwise differentiation.
     *
     * @param K Strike
     * @param T Time to expiry (years)
     * @return  Greeks struct
     */
    [[nodiscard]] Greeks pathwise(double K, double T) const;

    /**
     * @brief Compute Greeks via central finite differences (slow, for validation).
     *
     * Used in unit tests to verify the pathwise implementation.
     *
     * @param K Strike
     * @param T Expiry
     * @param h Perturbation size (default 1e-4)
     * @return  Greeks struct computed by finite differences
     */
    [[nodiscard]] Greeks finite_difference(double K, double T, double h = 1e-4) const;

    /**
     * @brief Compute pathwise Greeks for every option in mkt.
     * @return Vector of Greeks indexed identically to mkt.options
     */
    [[nodiscard]] std::vector<Greeks> compute_all(const MarketData& mkt) const;

private:
    HestonParams params_;
    MarketData   mkt_;

    // FFT config matching the main pricer
    static constexpr int    FFT_N   = 4096;
    static constexpr double FFT_ETA = 0.25;
    static constexpr double ALPHA   = 1.5;

    /**
     * @brief Compute call price and its derivative w.r.t. S via one FFT.
     *
     * Returns {price, delta} using pathwise differentiation.
     */
    [[nodiscard]] std::pair<double, double>
    price_and_delta(double K, double T) const;

    /**
     * @brief Compute call price and its derivative w.r.t. v₀ via one FFT.
     *
     * Returns {price, vega} using the identity ∂φ/∂v₀ = D·φ.
     */
    [[nodiscard]] std::pair<double, double>
    price_and_vega(double K, double T) const;
};

} // namespace heston
