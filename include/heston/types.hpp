#pragma once
/**
 * @file types.hpp
 * @brief Core data structures for the Heston calibration engine.
 *
 * Defines HestonParams (model parameters), OptionData (single contract),
 * MarketData (full option chain) and CalibrationResult (output record).
 */

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace heston {

// ---------------------------------------------------------------------------
// Model parameters
// ---------------------------------------------------------------------------

/**
 * @brief Five-parameter Heston (1993) stochastic-volatility model.
 *
 * Risk-neutral dynamics:
 *   dS = r S dt + sqrt(v) S dW1
 *   dv = kappa(theta - v) dt + xi sqrt(v) dW2
 *   E[dW1 dW2] = rho dt
 *
 * Feller condition: 2 kappa theta > xi^2
 *   When satisfied the CIR variance process never reaches zero.
 */
struct HestonParams {
    double v0;    ///< Initial variance (must be > 0)
    double kappa; ///< Mean-reversion speed (must be > 0)
    double theta; ///< Long-run variance (must be > 0)
    double xi;    ///< Vol-of-vol, sigma in many texts (must be > 0)
    double rho;   ///< Brownian correlation in (-1, 1)

    /// @brief Feller condition 2κθ > ξ².  When false, v can hit zero.
    [[nodiscard]] bool feller_satisfied() const noexcept {
        return 2.0 * kappa * theta > xi * xi;
    }

    /// @brief Parameter feasibility check.
    [[nodiscard]] bool valid() const noexcept {
        return v0 > 0.0 && kappa > 0.0 && theta > 0.0 && xi > 0.0
               && rho > -1.0 && rho < 1.0;
    }

    /// @brief Human-readable dump to stdout.
    void print() const {
        std::printf("HestonParams {\n");
        std::printf("  v0    = %.6f  (σ_0 = %.4f%%)\n", v0, 100.0 * std::sqrt(v0));
        std::printf("  kappa = %.6f\n", kappa);
        std::printf("  theta = %.6f  (σ_∞ = %.4f%%)\n", theta, 100.0 * std::sqrt(theta));
        std::printf("  xi    = %.6f\n", xi);
        std::printf("  rho   = %.6f\n", rho);
        std::printf("  Feller 2κθ > ξ²: %s  (2κθ=%.6f, ξ²=%.6f)\n",
                    feller_satisfied() ? "YES" : "NO",
                    2.0 * kappa * theta, xi * xi);
        std::printf("}\n");
    }
};

// ---------------------------------------------------------------------------
// Single option contract
// ---------------------------------------------------------------------------

/**
 * @brief Market data for one listed option contract.
 *
 * Prices are in the same currency units as the underlying.
 * Implied vol (market_iv) is the Black-Scholes σ that prices the mid quote.
 */
struct OptionData {
    double strike;        ///< Strike price K
    double expiry;        ///< Time to expiry T in years
    double market_price;  ///< Mid-market price (bid+ask)/2
    double market_iv;     ///< Black-Scholes implied vol from mid price
    double bid;
    double ask;
    double open_interest;
    int    type;          ///< 0 = call, 1 = put

    /// @brief Bid-ask spread.
    [[nodiscard]] double spread() const noexcept { return ask - bid; }

    /// @brief Moneyness K / F where F = spot * e^{(r-q)*T} (caller provides F).
    [[nodiscard]] double moneyness(double F) const noexcept {
        return (F > 0.0) ? strike / F : 0.0;
    }
};

// ---------------------------------------------------------------------------
// Full option chain
// ---------------------------------------------------------------------------

/**
 * @brief Aggregated market data: spot, rates and a vector of options.
 *
 * Liquidity filter removes wide-spread or low-OI contracts that would
 * corrupt the calibration objective.
 */
struct MarketData {
    double spot;      ///< Current underlier price S₀
    double rate;      ///< Continuously-compounded risk-free rate r
    double div_yield; ///< Continuous dividend yield q

    std::vector<OptionData> options;

    /**
     * @brief Return a copy containing only liquid options.
     *
     * Criteria:
     *   - open_interest > 100
     *   - spread / mid < 0.10  (< 10 % of mid price)
     *   - moneyness K/F in (0.70, 1.30)
     */
    [[nodiscard]] MarketData filter_liquid() const {
        MarketData out = *this;
        out.options.clear();
        for (const auto& opt : options) {
            double F = spot * std::exp((rate - div_yield) * opt.expiry);
            double mid = opt.market_price;
            if (mid <= 0.0) continue;
            double km = opt.strike / F;
            double spread_ratio = opt.spread() / mid;
            if (opt.open_interest > 100.0
                && spread_ratio < 0.10
                && km > 0.70 && km < 1.30) {
                out.options.push_back(opt);
            }
        }
        return out;
    }
};

// ---------------------------------------------------------------------------
// Calibration output
// ---------------------------------------------------------------------------

/**
 * @brief Full calibration output: parameters, errors, diagnostics.
 *
 * RMSE and max_error are expressed in implied-vol bps (1 bps = 0.01 %).
 * A production-quality fit achieves RMSE < 50 bps on liquid SPX options.
 */
struct CalibrationResult {
    HestonParams params;
    double rmse;          ///< Root mean squared IV error (bps)
    double max_error;     ///< Worst single-option IV error (bps)
    int    n_iterations;
    double runtime_ms;
    bool   converged;

    std::vector<double> model_ivs;  ///< Heston model IVs indexed as options
    std::vector<double> market_ivs; ///< Corresponding market IVs

    /// @brief Formatted summary dump.
    void print() const {
        std::printf("CalibrationResult {\n");
        params.print();
        std::printf("  RMSE          = %.2f bps\n", rmse * 1e4);
        std::printf("  Max error     = %.2f bps\n", max_error * 1e4);
        std::printf("  Iterations    = %d\n", n_iterations);
        std::printf("  Runtime       = %.1f ms\n", runtime_ms);
        std::printf("  Converged     = %s\n", converged ? "YES" : "NO");
        std::printf("  N options     = %zu\n", model_ivs.size());
        std::printf("}\n");
    }
};

} // namespace heston
