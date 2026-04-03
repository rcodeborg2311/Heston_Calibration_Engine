#pragma once
/**
 * @file vol_surface.hpp
 * @brief Implied-volatility surface: fit, interpolation and CSV export.
 *
 * Wraps the calibrated Heston pricer to produce a continuous vol surface
 * over an arbitrary (K, T) grid, supporting:
 *   - Dense grid evaluation for 3-D plotting
 *   - Cubic-spline interpolation in the moneyness/expiry dimensions
 *   - CSV export for use by plot_surface.py
 *   - Simple static-arbitrage checks (calendar spread, butterfly)
 */

#include "types.hpp"
#include <string>
#include <vector>

namespace heston {

/**
 * @brief Represents the implied-vol surface on a 2-D (K, T) grid.
 *
 * The surface is evaluated at a user-specified grid of strikes (as moneyness
 * K/S) and expiries, then stored as a matrix for efficient lookup.
 *
 * Interpolation between grid points uses bilinear interpolation (fast) or
 * bicubic interpolation (smoother — enabled by default).
 */
class VolSurface {
public:
    /**
     * @brief Grid configuration for surface evaluation.
     */
    struct Config {
        int    n_strikes   = 50;  ///< Number of strike grid points
        int    n_expiries  = 20;  ///< Number of expiry grid points
        double moneyness_lo = 0.70; ///< Minimum moneyness K/S
        double moneyness_hi = 1.30; ///< Maximum moneyness K/S
        double expiry_lo    = 0.05; ///< Minimum expiry (years, ~2 weeks)
        double expiry_hi    = 2.0;  ///< Maximum expiry (years)
    };

    /**
     * @brief Construct and evaluate the full surface.
     *
     * @param params Calibrated Heston parameters
     * @param mkt    Market data (spot, rate, yield)
     * @param cfg    Grid configuration
     */
    /// Construct with default Config (50 strikes, 20 expiries)
    VolSurface(const HestonParams& params, const MarketData& mkt);
    VolSurface(const HestonParams& params, const MarketData& mkt, Config cfg);

    /**
     * @brief Query model implied vol at any (K, T) via bilinear interpolation.
     * @param K Strike
     * @param T Expiry (years)
     * @return  Model implied vol (annualised)
     */
    [[nodiscard]] double implied_vol(double K, double T) const;

    /**
     * @brief Export surface and market data to CSV for Python plotting.
     *
     * Columns: strike, expiry, market_iv, model_iv
     *
     * market_iv is filled from the provided MarketData; NaN for grid points
     * that don't correspond to a traded option.
     *
     * @param path       Output file path (e.g., "results/calibration_output.csv")
     * @param market     Market data with observed IVs
     */
    void export_csv(const std::string& path, const MarketData& market) const;

    /**
     * @brief Check for static arbitrage on the fitted surface.
     *
     * Checks:
     *   1. Calendar spread: σ(K,T₁) ≤ σ(K,T₂) for T₁ < T₂ (not always required
     *      but a strong signal of fit quality)
     *   2. Butterfly (convexity in K): ∂²C/∂K² ≥ 0 (model prices must be convex)
     *
     * @return true if no arbitrage detected
     */
    [[nodiscard]] bool check_no_arbitrage() const;

    // Accessors for Python export
    [[nodiscard]] const std::vector<double>& strike_grid()  const { return K_grid_; }
    [[nodiscard]] const std::vector<double>& expiry_grid()  const { return T_grid_; }
    [[nodiscard]] const std::vector<std::vector<double>>& iv_matrix() const { return iv_; }

private:
    HestonParams params_;
    MarketData   mkt_;
    Config       cfg_;

    std::vector<double>              K_grid_; ///< Strike grid
    std::vector<double>              T_grid_; ///< Expiry grid
    std::vector<std::vector<double>> iv_;     ///< iv_[i][j] = IV at (K_i, T_j)

    /// Evaluate the full grid by calling CarrMadanPricer for each expiry slice.
    void build();
};

} // namespace heston
