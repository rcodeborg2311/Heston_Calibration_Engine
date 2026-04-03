#pragma once
/**
 * @file adi_solver.hpp
 * @brief Craig-Sneyd ADI finite-difference solver for the Heston PDE.
 *
 * Solves the backward Kolmogorov PDE derived from the Heston model:
 *
 *   ∂u/∂t + ½vS²∂²u/∂S² + ρξvS∂²u/∂S∂v + ½ξ²v∂²u/∂v²
 *           + (r-q)S∂u/∂S + κ(θ-v)∂u/∂v - ru = 0
 *
 * Terminal condition: u(T,S,v) = max(S−K, 0) for a European call.
 *
 * Spatial discretisation uses non-uniform grids concentrated near the
 * current spot S₀ and initial variance v₀ for accuracy with fewer points.
 *
 * Time discretisation uses the Craig-Sneyd (CS) ADI scheme which achieves
 * second-order accuracy in both space and time.  Reference:
 *   In 't Hout, K. J. & Foulon, S. (2010). "ADI finite difference schemes
 *   for option pricing in the Heston model with correlation,"
 *   Int. J. Numer. Anal. Model. 7(2), 303-320.
 */

#include "types.hpp"
#include <vector>

namespace heston {

/**
 * @brief Finite-difference solver for the 2-D Heston PDE.
 *
 * Uses the Craig-Sneyd splitting scheme with the Thomas algorithm for
 * the implicit sub-steps.  All workspace arrays are pre-allocated to
 * avoid heap allocation in the solve loop.
 */
class ADISolver {
public:
    /**
     * @brief Grid and time-stepping configuration.
     */
    struct Config {
        int    Ns         = 100;  ///< Number of S grid points
        int    Nv         = 50;   ///< Number of v grid points
        int    Nt         = 200;  ///< Number of time steps (backward)
        double Smax_mult  = 5.0;  ///< Smax = Smax_mult * S0
        double vmax_mult  = 5.0;  ///< vmax = vmax_mult * theta
        double c_s        = 3.0;  ///< S-grid concentration parameter
        double c_v        = 0.5;  ///< v-grid concentration parameter
        double theta_cs   = 0.5;  ///< CS blending factor (0.5 = Crank-Nicolson)
    };

    /**
     * @brief Output of a grid-convergence study.
     *
     * Prices at increasing Nt values should converge with O(Δt²) ≈ order 2.
     * Richardson extrapolation removes the leading error term.
     */
    struct ConvergenceResult {
        std::vector<int>    Nt_values;
        std::vector<double> prices;
        double              extrapolated_price;  ///< Richardson limit
        double              order_of_convergence; ///< Should be ~2 for CS scheme
    };

    /**
     * @brief Construct solver and pre-allocate all workspace.
     * @param params Heston parameters (used for coefficients)
     * @param mkt    Market data (spot S₀, rate r, yield q)
     * @param cfg    Grid configuration
     */
    /// Construct with default Config (Ns=100, Nv=50, Nt=200)
    ADISolver(const HestonParams& params, const MarketData& mkt);
    ADISolver(const HestonParams& params, const MarketData& mkt, Config cfg);

    /**
     * @brief Price a European call option via 2-D PDE solve.
     * @param K Strike price
     * @param T Time to expiry (years)
     * @return  Call price at (S₀, v₀)
     */
    [[nodiscard]] double call_price(double K, double T) const;

    /**
     * @brief Run a grid-convergence study at four Nt refinements.
     *
     * Computes prices at Nt = {50, 100, 200, 400}, fits the convergence order
     * and applies Richardson extrapolation for a high-accuracy reference value.
     *
     * @param K Strike
     * @param T Expiry
     * @return  Convergence results with extrapolated price
     */
    [[nodiscard]] ConvergenceResult convergence_study(double K, double T) const;

private:
    HestonParams params_;
    MarketData   mkt_;
    Config       cfg_;

    // Spatial grids (computed once)
    std::vector<double> S_grid_; ///< Ns points: S_i = S0*sinh(c_s*i/Ns)/sinh(c_s)
    std::vector<double> v_grid_; ///< Nv points: v_j = vmax*sinh(c_v*j/Nv)/sinh(c_v)

    // ---------------------------------------------------------------
    // Core solve — separate from call_price so convergence_study can
    // call it with different Nt without rebuilding grids.
    // ---------------------------------------------------------------

    /**
     * @brief Full backward-in-time PDE solve.
     *
     * Fills U[Ns×Nv] with option values at t=0.  Implements the six-step
     * Craig-Sneyd scheme of In 't Hout & Foulon (2010), Algorithm 1.
     *
     * @param K  Strike
     * @param T  Expiry
     * @param Nt Number of time steps to use (overrides cfg_.Nt)
     * @return   Call price interpolated to (S₀, v₀)
     */
    [[nodiscard]] double solve(double K, double T, int Nt) const;

    // Grid construction
    void build_grids();

    // Terminal and boundary conditions
    void apply_terminal_condition(double K, std::vector<double>& U) const;
    void apply_boundary_conditions(std::vector<double>& U,
                                   double K, double r, double q,
                                   double tau) const;

    // Bilinear interpolation to (S₀, v₀)
    [[nodiscard]] double interpolate_to_spot(const std::vector<double>& U) const;

    /**
     * @brief Thomas algorithm (tridiagonal matrix algorithm) in-place.
     *
     * Solves A·x = d where A is tridiagonal with:
     *   a[i] = subdiagonal (a[0] unused)
     *   b[i] = main diagonal
     *   c[i] = superdiagonal (c[n-1] unused)
     *   d[i] = right-hand side (overwritten with solution)
     *
     * O(n) flops, no heap allocation.
     *
     * @param a  Subdiagonal
     * @param b  Main diagonal (modified in place — workspace)
     * @param c  Superdiagonal
     * @param d  RHS on entry, solution on exit
     */
    static void thomas_solve(std::vector<double>& a,
                             std::vector<double>& b,
                             std::vector<double>& c,
                             std::vector<double>& d) noexcept;
};

} // namespace heston
