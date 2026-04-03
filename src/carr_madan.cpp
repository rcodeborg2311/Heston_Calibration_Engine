/**
 * @file carr_madan.cpp
 * @brief Carr-Madan (1999) FFT option pricing implementation.
 *
 * Reference:
 *   Carr, P. & Madan, D. (1999). "Option Valuation Using the Fast Fourier
 *   Transform." Journal of Computational Finance 2(4), 61-73.
 *
 * The dampening trick (Carr-Madan §2):
 *   Define c_T(k) = e^{αk} C(e^k),  k = ln(K/F)
 *   Then c_T is square-integrable for α > 0 (with α < critical moment).
 *   Its Fourier transform is:
 *
 *     ψ(v) = e^{-rT} φ(v − (α+1)i) / (α² + α − v² + i(2α+1)v)
 *
 *   and the call price is recovered by:
 *     C(K) = (e^{-αk}/π) Re[∫₀^∞ e^{ivk} ψ(v) dv]
 *
 *   Discretised with N points, step η in v-space:
 *     v_j = η·j,  j = 0,...,N-1
 *   Corresponding log-strike grid with step λ = 2π/(Nη):
 *     k_u = −λN/2 + λu  (centred at k=0 i.e. ATM)
 *
 * Simpson's rule weights improve accuracy from O(η²) to O(η⁴):
 *   w_j = (η/3)·{1, 4, 2, 4, 2, ..., 4, 1}
 *
 * Note: FFTW is called with FFTW_MEASURE on the first construction.
 * The plan is reused for all subsequent FFT calls.
 */

#include "heston/carr_madan.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace heston {

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

CarrMadanPricer::CarrMadanPricer(const HestonCharacteristicFn& cf)
    : CarrMadanPricer(cf, Config{})
{}

CarrMadanPricer::CarrMadanPricer(const HestonCharacteristicFn& cf, Config cfg)
    : cf_(cf), cfg_(cfg)
{
    const int N = cfg_.N;
    fftw_in_  = fftw_alloc_complex(static_cast<std::size_t>(N));
    fftw_out_ = fftw_alloc_complex(static_cast<std::size_t>(N));
    if (!fftw_in_ || !fftw_out_) {
        throw std::runtime_error("CarrMadanPricer: FFTW allocation failed");
    }
    // Carr-Madan (1999) eq. 9: C_T(k) = (e^{-αk}/π) Re[∫₀^∞ e^{-ivk} ψ(v) dv]
    // The exp(-ivk) integral matches the FORWARD DFT convention
    // (FFTW_FORWARD computes Σ_j x[j] exp(-2πijk/N)).
    // A shift factor exp(+ivb) in the input moves the log-strike grid to centre.
    plan_ = fftw_plan_dft_1d(N, fftw_in_, fftw_out_, FFTW_FORWARD, FFTW_MEASURE);
    if (!plan_) {
        throw std::runtime_error("CarrMadanPricer: FFTW plan creation failed");
    }

    log_strike_grid_.resize(static_cast<std::size_t>(N));
    call_prices_fft_.resize(static_cast<std::size_t>(N));
}

CarrMadanPricer::~CarrMadanPricer()
{
    if (plan_)    fftw_destroy_plan(plan_);
    if (fftw_in_)  fftw_free(fftw_in_);
    if (fftw_out_) fftw_free(fftw_out_);
}

// ---------------------------------------------------------------------------
// Internal FFT computation
//
// Fills log_strike_grid_ and call_prices_fft_.
// One call per expiry; O(N log N) work.
// ---------------------------------------------------------------------------

void CarrMadanPricer::run_fft(double T, double F) const
{
    const int    N      = cfg_.N;
    const double eta    = cfg_.eta;
    const double alpha  = cfg_.alpha;
    const double lambda = cfg_.lambda();
    const double r      = cf_.r();

    // ln(F) = log of forward price; log-strike grid is centred here (ATM)
    const double ln_F = std::log(F);

    // -----------------------------------------------------------------------
    // Carr-Madan (1999) §2, eqs. 9-11.
    //
    // Log-strike convention: k = ln(K)  (absolute level, not moneyness).
    // The CF used is φ_T(u) = E^Q[e^{iu·ln(S_T)}]  — full drift included.
    //
    // ψ(v) = e^{-rT} · φ_T(v − (α+1)i; T) / (α²+α−v²+i(2α+1)v)
    //
    // C(K) = (K^{-α}/π) · Re[∫₀^∞ e^{−ivk} ψ(v) dv]   with k = ln(K)
    //
    // FFT discretisation (N points, step η, FFTW_FORWARD = exp(−2πijk/N)):
    //   k_u = ln(F) + (u − N/2)·λ   (centred at ATM)
    //   x[j] = w_j · ψ(v_j) · exp(−iv_j · ln(F)) · (−1)^j
    //   X[u] = Σ_j x[j] · exp(−2πiju/N)
    //        = Σ_j w_j ψ_j exp(−iv_j k_u)    ✓
    //
    // Because: −iv_j·ln(F) + iπj − 2πiju/N
    //        = −iv_j·(ln(F) + 2πu/(Nη) − π/η)
    //        = −iv_j·k_u  (when λ = 2π/(Nη))                      [QED]
    // -----------------------------------------------------------------------

    using cd = std::complex<double>;

    for (int j = 0; j < N; ++j) {
        const double v_j = static_cast<double>(j) * eta;

        // Dampened integration variable: u = v_j − (α+1)·i
        const cd u = cd(v_j, -(alpha + 1.0));

        // Full CF of ln(S_T)  [Albrecher et al. 2007, eq. 4-5]
        const cd phi = cf_(u, T);

        // Denominator: α²+α−v²+i(2α+1)v  [Carr-Madan 1999, eq. 11]
        const cd denom = cd(alpha*alpha + alpha - v_j*v_j,
                            (2.0*alpha + 1.0)*v_j);

        // ψ(v_j) = e^{-rT} · φ(u) / denom
        const cd psi = std::exp(-r * T) * phi / denom;

        // Simpson's rule weight w_j
        double w = 1.0;
        if (cfg_.use_simpsons) {
            if      (j == 0 || j == N-1) w = 1.0;
            else if (j % 2 == 1)         w = 4.0;
            else                         w = 2.0;
            w *= eta / 3.0;
        } else {
            w = eta;
        }

        // Phase: exp(−iv_j · ln(F)) · (−1)^j  — centres grid at ATM
        const cd phase = std::exp(cd(0.0, -v_j * ln_F))
                         * cd(((j % 2 == 0) ? 1.0 : -1.0), 0.0);

        const cd val = cd(w, 0.0) * psi * phase;
        fftw_in_[j][0] = val.real();
        fftw_in_[j][1] = val.imag();
    }

    // Execute FFT (FFTW_FORWARD: Σ x[j] exp(−2πiju/N))
    fftw_execute(plan_);

    // Extract log-strike grid and call prices
    // k_u = ln(F) + (u − N/2)·λ  →  K_u = F · exp((u − N/2)·λ)
    // C(K_u) = (K_u^{−α}/π) · Re[X[u]]
    const double b = lambda * static_cast<double>(N) / 2.0;  // half-width
    for (int u = 0; u < N; ++u) {
        const double k_u = ln_F - b + lambda * static_cast<double>(u);
        const double re_fft = fftw_out_[u][0];

        // Undampened call price: C = exp(−α·k) / π · Re[FFT]
        //   = K^{−α} / π · Re[FFT]
        const double C_u = (std::exp(-alpha * k_u) / M_PI) * re_fft;

        log_strike_grid_[static_cast<std::size_t>(u)] = k_u;
        call_prices_fft_[static_cast<std::size_t>(u)] = C_u;
    }
}

// ---------------------------------------------------------------------------
// Cubic-spline interpolation in log-strike space
// ---------------------------------------------------------------------------

double CarrMadanPricer::interpolate_call(double k) const
{
    const int N = cfg_.N;

    // Find the interval [k_j, k_{j+1}] containing k
    const auto it = std::lower_bound(log_strike_grid_.begin(),
                                     log_strike_grid_.end(), k);
    if (it == log_strike_grid_.end()) {
        return call_prices_fft_.back();
    }
    if (it == log_strike_grid_.begin()) {
        return call_prices_fft_.front();
    }

    const std::size_t j = static_cast<std::size_t>(it - log_strike_grid_.begin());
    const std::size_t jm1 = j - 1;
    const std::size_t jp1 = std::min(j + 1, static_cast<std::size_t>(N - 1));
    const std::size_t jm2 = (jm1 > 0) ? jm1 - 1 : 0;

    // Cubic Catmull-Rom interpolation using four surrounding points
    const double k0 = log_strike_grid_[jm2];
    const double k1 = log_strike_grid_[jm1];
    const double k2 = log_strike_grid_[j];
    const double k3 = log_strike_grid_[jp1];

    const double p0 = call_prices_fft_[jm2];
    const double p1 = call_prices_fft_[jm1];
    const double p2 = call_prices_fft_[j];
    const double p3 = call_prices_fft_[jp1];

    // Local parameter t in [0,1]
    const double h = k2 - k1;
    const double t = (h > 0.0) ? (k - k1) / h : 0.0;
    const double t2 = t * t;
    const double t3 = t2 * t;

    // Catmull-Rom weights (tension = 0.5)
    const double hkm2 = k1 - k0;
    const double hkp1 = k3 - k2;
    const double m1 = (hkm2 > 0.0) ? 0.5 * (p2 - p0) / (h + hkm2) * 2.0 * h : (p2 - p1);
    const double m2 = (hkp1 > 0.0) ? 0.5 * (p3 - p1) / (h + hkp1) * 2.0 * h : (p2 - p1);

    // Cubic Hermite
    const double result = (2*t3 - 3*t2 + 1) * p1
                        + (t3 - 2*t2 + t)   * m1
                        + (-2*t3 + 3*t2)    * p2
                        + (t3 - t2)         * m2;
    return std::max(result, 0.0);
}

// ---------------------------------------------------------------------------
// Single call price
// ---------------------------------------------------------------------------

double CarrMadanPricer::call_price(double K, double T) const
{
    const double r = cf_.r();
    const double q = cf_.q();
    const double S = cf_.S0();
    const double F = S * std::exp((r - q) * T);

    run_fft(T, F);

    // k = ln(K)  (absolute log-strike, matching the FFT grid convention)
    const double k = std::log(K);
    return interpolate_call(k);
}

// ---------------------------------------------------------------------------
// Batch pricing: one FFT per unique expiry
// ---------------------------------------------------------------------------

std::vector<double> CarrMadanPricer::price_all(const MarketData& mkt) const
{
    const double r = cf_.r();
    const double q = cf_.q();
    const double S = cf_.S0();

    const std::size_t n = mkt.options.size();
    std::vector<double> prices(n, 0.0);

    // Group option indices by expiry
    std::map<double, std::vector<std::size_t>> by_expiry;
    for (std::size_t i = 0; i < n; ++i) {
        by_expiry[mkt.options[i].expiry].push_back(i);
    }

    for (const auto& [T, indices] : by_expiry) {
        const double F = S * std::exp((r - q) * T);
        run_fft(T, F);

        for (const std::size_t idx : indices) {
            const OptionData& opt = mkt.options[idx];
            double K = opt.strike;
            double k = std::log(K);  // absolute log-strike (Carr-Madan convention)
            double call = interpolate_call(k);

            if (opt.type == 1) {
                // Put via put-call parity: P = C - S*e^{-qT} + K*e^{-rT}
                call = call - S * std::exp(-q * T) + K * std::exp(-r * T);
                call = std::max(call, 0.0);
            }

            prices[idx] = call;
        }
    }

    return prices;
}

// ---------------------------------------------------------------------------
// Implied volatility inversion
// ---------------------------------------------------------------------------

// Black-Scholes call price
double CarrMadanPricer::bs_call(double S, double K, double T,
                                 double r, double q, double sigma) noexcept
{
    if (T <= 0.0 || sigma <= 0.0) return std::max(S * std::exp(-q * T) - K * std::exp(-r * T), 0.0);
    const double sqrtT = std::sqrt(T);
    const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;

    // Standard normal CDF via erfc
    auto N_cdf = [](double x) -> double {
        return 0.5 * std::erfc(-x * M_SQRT1_2);
    };

    return S * std::exp(-q * T) * N_cdf(d1) - K * std::exp(-r * T) * N_cdf(d2);
}

// Black-Scholes vega
double CarrMadanPricer::bs_vega(double S, double K, double T,
                                 double r, double q, double sigma) noexcept
{
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    const double sqrtT = std::sqrt(T);
    const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    // φ(d1) = exp(-d1^2/2)/sqrt(2π)
    const double phi_d1 = std::exp(-0.5 * d1 * d1) * M_SQRT1_2 * M_2_SQRTPI * 0.5;
    return S * std::exp(-q * T) * phi_d1 * sqrtT;
}

// Brent's method for IV inversion (guaranteed convergence when bracketed)
double CarrMadanPricer::brent_iv(double target, double S, double K, double T,
                                  double r, double q,
                                  double tol, int max_iter) noexcept
{
    auto f = [&](double sigma) {
        return bs_call(S, K, T, r, q, sigma) - target;
    };

    double a = 1e-6, b = 10.0;
    double fa = f(a), fb = f(b);

    if (fa * fb > 0.0) {
        // Can't bracket — intrinsic value check
        return std::numeric_limits<double>::quiet_NaN();
    }

    double c = a, fc = fa;
    bool mflag = true;
    double s = 0.0, d = 0.0;

    for (int i = 0; i < max_iter; ++i) {
        if (std::abs(b - a) < tol) break;

        if (fa != fc && fb != fc) {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc))
              + b * fa * fc / ((fb - fa) * (fb - fc))
              + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant
            s = b - fb * (b - a) / (fb - fa);
        }

        const double cond1 = ((s < (3 * a + b) / 4.0) || (s > b));
        const double cond2 = (mflag  && std::abs(s - b) >= std::abs(b - c) / 2.0);
        const double cond3 = (!mflag && std::abs(s - b) >= std::abs(c - d) / 2.0);
        const double cond4 = (mflag  && std::abs(b - c) < tol);
        const double cond5 = (!mflag && std::abs(c - d) < tol);

        if (cond1 || cond2 || cond3 || cond4 || cond5) {
            s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }

        const double fs = f(s);
        d = c; c = b; fc = fb;

        if (fa * fs < 0.0) { b = s; fb = fs; }
        else                { a = s; fa = fs; }

        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b); std::swap(fa, fb);
        }
    }

    return b;
}

// Newton-Raphson with Brent fallback
double CarrMadanPricer::implied_vol(double target, double S, double K, double T,
                                     double r, double q,
                                     double tol, int max_iter) noexcept
{
    if (target <= 0.0) return std::numeric_limits<double>::quiet_NaN();

    // Intrinsic value check
    const double intrinsic = std::max(S * std::exp(-q * T) - K * std::exp(-r * T), 0.0);
    if (target < intrinsic - 1e-6) return std::numeric_limits<double>::quiet_NaN();

    // Newton-Raphson
    double sigma = 0.2; // starting guess ~20%
    for (int i = 0; i < max_iter; ++i) {
        const double price = bs_call(S, K, T, r, q, sigma);
        const double vega  = bs_vega(S, K, T, r, q, sigma);
        if (vega < 1e-12) break;

        const double diff = price - target;
        if (std::abs(diff) < tol) return sigma;

        sigma -= diff / vega;
        if (sigma < 1e-6 || sigma > 10.0) break;
    }

    // Fallback to Brent's method
    return brent_iv(target, S, K, T, r, q, tol, max_iter);
}

// Implied vols for all options
std::vector<double> CarrMadanPricer::implied_vols(const MarketData& mkt) const
{
    const double S = cf_.S0();
    const double r = cf_.r();
    const double q = cf_.q();

    std::vector<double> model_prices = price_all(mkt);
    std::vector<double> ivs;
    ivs.reserve(mkt.options.size());

    for (std::size_t i = 0; i < mkt.options.size(); ++i) {
        const OptionData& opt = mkt.options[i];
        double price = model_prices[i];
        // For puts, convert to call equivalent for IV computation
        if (opt.type == 1) {
            // Call = Put + S*e^{-qT} - K*e^{-rT}
            price = price + S * std::exp(-q * opt.expiry) - opt.strike * std::exp(-r * opt.expiry);
            price = std::max(price, 0.0);
        }
        ivs.push_back(implied_vol(price, S, opt.strike, opt.expiry, r, q));
    }

    return ivs;
}

// ---------------------------------------------------------------------------
// Benchmark: median of 100 FFT calls
// ---------------------------------------------------------------------------

double CarrMadanPricer::benchmark_us() const
{
    const double T = 0.25;
    const double F = cf_.S0() * std::exp((cf_.r() - cf_.q()) * T);

    std::vector<double> timings(100);
    for (int i = 0; i < 100; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        run_fft(T, F);
        const auto t1 = std::chrono::high_resolution_clock::now();
        timings[static_cast<std::size_t>(i)] =
            std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    std::sort(timings.begin(), timings.end());
    return timings[50];
}

} // namespace heston
