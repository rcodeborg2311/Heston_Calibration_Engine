#pragma once
/**
 * @file calibrator.hpp
 * @brief Two-phase Heston model calibration engine.
 *
 * Phase 1 — Differential Evolution (DE): global stochastic search that finds
 *   the basin of attraction for the optimal parameters.
 *
 * Phase 2 — Levenberg-Marquardt (LM): gradient-based local refinement that
 *   polishes the DE solution to high accuracy.
 *
 * Objective: minimise RMSE of Black-Scholes implied volatilities
 *   L(Θ) = √(1/N · Σᵢ wᵢ [σ_model(Kᵢ,Tᵢ;Θ) − σ_market(Kᵢ,Tᵢ)]²)
 *   with weights wᵢ = 1 / bid_ask_spread (tighter spread → more weight).
 *
 * Parameter constraints:
 *   v₀ ∈ [1e-6, 1.0]    κ ∈ [0.01, 20.0]
 *   θ ∈ [1e-6, 1.0]     ξ ∈ [0.01, 5.0]
 *   ρ ∈ (−0.999, 0.999)
 */

#include "types.hpp"
#include <array>
#include <random>

namespace heston {

/**
 * @brief Heston model calibration engine.
 *
 * Thread safety: Calibrator is not thread-safe.  Use separate instances
 * for parallel calibration runs.
 */
class Calibrator {
public:
    /**
     * @brief Calibration hyper-parameters.
     */
    struct Config {
        int    max_iter_lm   = 500;   ///< Max Levenberg-Marquardt iterations
        int    max_iter_de   = 1000;  ///< Max Differential Evolution generations
        int    de_population = 15;    ///< DE population size (10-15 recommended)
        double de_F          = 0.8;   ///< DE mutation scale factor F ∈ (0,2)
        double de_CR         = 0.9;   ///< DE crossover probability CR ∈ (0,1)
        double tol           = 1e-6;  ///< Convergence tolerance on loss
        bool   use_weights   = true;  ///< Weight options by inverse bid-ask spread
        int    n_starts      = 5;     ///< Multi-start LM restarts from random init
        bool   verbose       = false; ///< Print iteration progress
        unsigned seed        = 42u;   ///< RNG seed for reproducibility
    };

    /// Construct with default Config
    Calibrator();
    explicit Calibrator(Config cfg);

    /**
     * @brief Full two-phase calibration: DE global search + LM refinement.
     *
     * This is the primary method.
     *   1. DE explores the full 5-D parameter space (max_iter_de generations).
     *   2. LM refines from the best DE solution (max_iter_lm iterations).
     *
     * @param mkt Market data (should be pre-filtered with filter_liquid())
     * @return    CalibrationResult with parameters, errors, diagnostics
     */
    [[nodiscard]] CalibrationResult calibrate(const MarketData& mkt);

    /**
     * @brief Single-phase local calibration from a given starting point.
     *
     * Faster than the two-phase approach — useful when:
     *   - A good initial guess is available (e.g., from a prior day's calibration)
     *   - Running real-time recalibration when market moves slightly
     *
     * @param mkt           Market data
     * @param initial_guess Starting parameter vector
     * @return              CalibrationResult
     */
    [[nodiscard]] CalibrationResult calibrate_local(const MarketData& mkt,
                                                     const HestonParams& initial_guess);

    /**
     * @brief Validate calibration quality and print a report.
     *
     * Checks:
     *   - RMSE < 50 bps (production bar for liquid SPX options)
     *   - Max error < 100 bps
     *   - Feller condition
     *   - Parameter bounds
     *
     * @param result CalibrationResult to inspect
     */
    static void validate(const CalibrationResult& result);

    // Parameter bounds (public for testing)
    static constexpr double V0_MIN = 1e-6, V0_MAX = 1.0;
    static constexpr double K_MIN  = 0.01,  K_MAX  = 20.0;   // kappa
    static constexpr double TH_MIN = 1e-6, TH_MAX = 1.0;     // theta
    static constexpr double XI_MIN = 0.01,  XI_MAX = 5.0;
    static constexpr double RH_MIN = -0.999, RH_MAX = 0.999;  // rho

private:
    Config      cfg_;
    std::mt19937 rng_;

    // Parameter vector encoding: theta = {v0, kappa, theta, xi, rho}
    using ParamVec = std::array<double, 5>;

    static HestonParams to_params(const ParamVec& x) noexcept;
    static ParamVec     to_vec(const HestonParams& p) noexcept;

    // -------------------------------------------------------------------
    // Differential Evolution
    // -------------------------------------------------------------------

    /**
     * @brief DE/rand/1/bin variant.
     *
     * Standard DE from Storn & Price (1997):
     *   mutation:    v = x_r1 + F·(x_r2 − x_r3)
     *   crossover:   u_j = v_j if U(0,1)<CR else x_j
     *   selection:   x ← u if f(u) ≤ f(x)
     *
     * @param mkt    Market data for objective evaluation
     * @return       Best parameter vector found
     */
    [[nodiscard]] HestonParams differential_evolution(const MarketData& mkt);

    // -------------------------------------------------------------------
    // Levenberg-Marquardt
    // -------------------------------------------------------------------

    /**
     * @brief LM iteration step.
     *
     * Jacobian is computed via central finite differences with step h = 1e-5.
     * The normal equations (J^T J + λ I) δ = J^T r are solved by Cholesky
     * (via a simple 5×5 system that can be solved directly).
     *
     * @param current     Current parameter vector
     * @param mkt         Market data
     * @param pricer      FFT pricer (pre-constructed for current params)
     * @param lambda_lm   Marquardt damping parameter λ
     * @param current_loss Current residual sum (updated on output)
     * @return            Updated parameter vector
     */
    HestonParams lm_step(const HestonParams& current,
                          const MarketData& mkt,
                          double lambda_lm,
                          double& current_loss) const;

    // -------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------

    /**
     * @brief Project parameter vector back into the feasible box.
     *
     * Called after every gradient step to enforce constraints.
     */
    static HestonParams project_constraints(const HestonParams& p) noexcept;

    /// Log a warning if the Feller condition is violated.
    static void check_feller(const HestonParams& p) noexcept;

    /**
     * @brief Compute the weighted IV-RMSE loss.
     *
     * @param p      Candidate parameters
     * @param mkt    Market data
     * @return       RMSE in implied vol (fractional, not bps)
     */
    [[nodiscard]] double evaluate_loss(const HestonParams& p,
                                        const MarketData& mkt) const;

    /**
     * @brief Compute residual vector r[i] = w_i * (σ_model[i] − σ_market[i]).
     *
     * The loss is ‖r‖₂ / √N.
     */
    [[nodiscard]] std::vector<double> residuals(const HestonParams& p,
                                                 const MarketData& mkt) const;

    /**
     * @brief Compute weights w[i] = 1 / spread[i] (normalised to sum to N).
     */
    [[nodiscard]] std::vector<double> compute_weights(const MarketData& mkt) const;

    /// Random parameter vector uniformly sampled from the feasible box.
    [[nodiscard]] HestonParams random_params();
};

} // namespace heston
