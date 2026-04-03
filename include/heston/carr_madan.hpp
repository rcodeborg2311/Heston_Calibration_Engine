#pragma once
/**
 * @file carr_madan.hpp
 * @brief Carr-Madan (1999) FFT option pricer for the Heston model.
 *
 * Implements the FFT-based European call pricing formula from:
 *   Carr, P. & Madan, D. (1999). "Option Valuation Using the Fast Fourier
 *   Transform." Journal of Computational Finance 2(4), 61-73.
 *
 * Key idea: by applying a dampening factor e^{αk} (α > 0) to the call price
 * as a function of log-strike k = ln(K/F), the result becomes square-integrable
 * and its Fourier transform can be computed via FFT in O(N log N) time,
 * pricing N strikes simultaneously.
 */

#include "characteristic_fn.hpp"
#include "types.hpp"
#include <fftw3.h>
#include <memory>
#include <vector>

namespace heston {

/**
 * @brief FFT call pricer using the Carr-Madan (1999) framework.
 *
 * Usage pattern (hot path):
 *   1. Construct once per parameter set (this creates the FFTW plan).
 *   2. Call price_all() or implied_vols() repeatedly — reuses the plan.
 *   3. Destroy to release FFTW resources.
 *
 * Thread safety: not thread-safe (FFTW plans are not reentrant).
 * For parallel calibration, construct one instance per thread.
 */
class CarrMadanPricer {
public:
    /**
     * @brief FFT configuration parameters.
     *
     * N must be a power of two for FFTW efficiency.
     * eta controls the integration step in frequency space; smaller eta
     * gives denser coverage in frequency but coarser strike spacing.
     * lambda = 2π/(N·η) is the resulting log-strike spacing.
     */
    struct Config {
        int    N           = 4096;    ///< FFT grid size (power of 2)
        double eta         = 0.25;    ///< Integration step in v-space (frequency)
        double alpha       = 1.5;     ///< Dampening exponent α (must be > 0)
        bool   use_simpsons = true;   ///< Simpson's rule weights (vs rectangular)

        /// Derived: log-strike grid spacing λ = 2π / (N·η)
        [[nodiscard]] double lambda() const noexcept {
            return 2.0 * M_PI / (static_cast<double>(N) * eta);
        }
    };

    /**
     * @brief Construct pricer.  Creates FFTW plan (expensive — do this once).
     * @param cf  Characteristic function to use (must outlive this object)
     * @param cfg FFT configuration
     */
    /// Construct with default Config (N=4096, eta=0.25, alpha=1.5)
    explicit CarrMadanPricer(const HestonCharacteristicFn& cf);
    CarrMadanPricer(const HestonCharacteristicFn& cf, Config cfg);
    ~CarrMadanPricer();

    // Non-copyable (owns FFTW plan pointer)
    CarrMadanPricer(const CarrMadanPricer&)            = delete;
    CarrMadanPricer& operator=(const CarrMadanPricer&) = delete;

    /**
     * @brief Price a single European call at (K, T).
     *
     * Runs a full N-point FFT and interpolates to the requested strike.
     * If pricing many strikes at the same expiry, prefer price_all().
     *
     * @param K Strike
     * @param T Time to expiry (years)
     * @return  European call price
     */
    [[nodiscard]] double call_price(double K, double T) const;

    /**
     * @brief Price all options in mkt simultaneously.
     *
     * Groups options by expiry to minimise the number of FFT calls.
     * One FFT call per unique expiry.
     *
     * @param mkt Market data (spot, rate, div yield, options list)
     * @return    Model call prices indexed identically to mkt.options
     */
    [[nodiscard]] std::vector<double> price_all(const MarketData& mkt) const;

    /**
     * @brief Compute Black-Scholes implied vols for all options in mkt.
     *
     * Steps:
     *   1. price_all() → model call prices (possibly converting puts via P-C parity)
     *   2. Newton inversion → BS implied vol
     *   3. Fall back to Brent's method if Newton diverges
     *
     * @param mkt Market data
     * @return    Implied vols indexed identically to mkt.options
     */
    [[nodiscard]] std::vector<double> implied_vols(const MarketData& mkt) const;

    /**
     * @brief Time one FFT call at configured N.
     * @return Elapsed time in microseconds (wall clock, median of 100 runs)
     */
    [[nodiscard]] double benchmark_us() const;

    // Expose market and characteristic function references
    [[nodiscard]] const HestonCharacteristicFn& cf()  const noexcept { return cf_; }
    [[nodiscard]] const Config&                 cfg() const noexcept { return cfg_; }

    // -----------------------------------------------------------------
    // Black-Scholes helpers (public for testing)
    // -----------------------------------------------------------------

    /**
     * @brief Black-Scholes call price.
     *
     * C = S e^{-qT} N(d1) - K e^{-rT} N(d2)
     * d1 = [ln(S/K) + (r-q+σ²/2)T] / (σ√T)
     * d2 = d1 - σ√T
     */
    static double bs_call(double S, double K, double T,
                          double r, double q, double sigma) noexcept;

    /**
     * @brief Black-Scholes vega ∂C/∂σ = S e^{-qT} φ(d1) √T.
     */
    static double bs_vega(double S, double K, double T,
                          double r, double q, double sigma) noexcept;

    /**
     * @brief Black-Scholes implied vol via Newton-Raphson with Brent fallback.
     *
     * @param target_price Market call price
     * @param S, K, T, r, q Standard BS inputs
     * @param tol      Convergence tolerance on price
     * @param max_iter Maximum Newton iterations before switching to Brent
     * @return         Implied vol (annualised), or NaN if inversion fails
     */
    static double implied_vol(double target_price, double S, double K, double T,
                              double r, double q,
                              double tol = 1e-8, int max_iter = 100) noexcept;

private:
    const HestonCharacteristicFn& cf_;
    Config cfg_;

    // FFTW resources (allocated once)
    fftw_plan        plan_{nullptr};
    fftw_complex*    fftw_in_{nullptr};
    fftw_complex*    fftw_out_{nullptr};

    // Pre-computed FFT inputs for one expiry (reused across strikes)
    mutable std::vector<double> log_strike_grid_; ///< k_j = ln(K_j / F)
    mutable std::vector<double> call_prices_fft_; ///< C_j from FFT output

    /**
     * @brief Run the FFT and populate call_prices_fft_ for one expiry.
     * @param T   Time to expiry
     * @param F   Forward price S * e^{(r-q)*T}
     */
    void run_fft(double T, double F) const;

    /**
     * @brief Cubic-spline interpolate call price at log-strike k.
     *
     * Uses the results of the most recent run_fft() call.
     */
    double interpolate_call(double k) const;

    /**
     * @brief Brent's method for BS IV inversion.
     *
     * Guaranteed convergence if a bracket [σ_lo, σ_hi] is found.
     */
    static double brent_iv(double target_price, double S, double K, double T,
                            double r, double q,
                            double tol, int max_iter) noexcept;
};

} // namespace heston
