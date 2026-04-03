/**
 * @file bench_pricing.cpp
 * @brief Nanobench performance benchmarks for the Heston pricing engine.
 *
 * Compares:
 *   1. Black-Scholes closed-form (baseline)
 *   2. Carr-Madan FFT (N=4096, one expiry)
 *   3. ADI finite-difference PDE (Ns=100, Nv=50)
 *   4. Full calibration (DE + LM, 30 options)
 */

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include "heston/adi_solver.hpp"
#include "heston/calibrator.hpp"
#include "heston/carr_madan.hpp"
#include "heston/characteristic_fn.hpp"
#include "heston/types.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace heston;
using namespace ankerl::nanobench;

static MarketData make_mkt(double S = 100.0, double r = 0.05, double q = 0.02)
{
    MarketData m;
    m.spot = S; m.rate = r; m.div_yield = q;
    return m;
}

static HestonParams bench_params()
{
    return {.v0 = 0.04, .kappa = 1.5, .theta = 0.04, .xi = 0.3, .rho = -0.7};
}

// Generate synthetic calibration data
static MarketData make_calib_data(const HestonParams& p, int n = 30)
{
    const double S = 100.0, r = 0.05, q = 0.02;
    MarketData mkt;
    mkt.spot = S; mkt.rate = r; mkt.div_yield = q;

    HestonCharacteristicFn cf(p, mkt);
    CarrMadanPricer pricer(cf);

    const std::vector<double> expiries = {0.25, 0.5, 1.0};
    int per_expiry = n / static_cast<int>(expiries.size());

    for (double T : expiries) {
        const double F = S * std::exp((r - q) * T);
        for (int i = 0; i < per_expiry; ++i) {
            double K = F * (0.85 + 0.30 * static_cast<double>(i) / (per_expiry - 1));
            double price = pricer.call_price(K, T);
            double iv = CarrMadanPricer::implied_vol(price, S, K, T, r, q);
            if (!std::isfinite(iv) || iv <= 0.0) continue;

            OptionData od{};
            od.strike = K; od.expiry = T;
            od.market_price = price; od.market_iv = iv;
            od.bid = price * 0.99; od.ask = price * 1.01;
            od.open_interest = 500; od.type = 0;
            mkt.options.push_back(od);
        }
    }
    return mkt;
}

int main()
{
    std::printf("\n====================================================================\n");
    std::printf("  Heston Engine Benchmark Results\n");
    std::printf("====================================================================\n\n");

    const HestonParams p = bench_params();
    const double S = 100.0, K = 100.0, T = 0.5, r = 0.05, q = 0.02;
    MarketData mkt = make_mkt(S, r, q);

    Bench bench;
    bench.warmup(10).minEpochIterations(100);

    // -----------------------------------------------------------------------
    // 1. Black-Scholes (baseline)
    // -----------------------------------------------------------------------

    double volatile bs_result = 0.0;
    bench.run("Black-Scholes (closed-form)", [&] {
        bs_result = CarrMadanPricer::bs_call(S, K, T, r, q, 0.20);
        ankerl::nanobench::doNotOptimizeAway(bs_result);
    });

    // -----------------------------------------------------------------------
    // 2. Carr-Madan FFT (N=4096)
    // -----------------------------------------------------------------------

    HestonCharacteristicFn cf(p, mkt);
    CarrMadanPricer pricer(cf);

    double volatile fft_result = 0.0;
    bench.run("Carr-Madan FFT (N=4096, all strikes)", [&] {
        fft_result = pricer.call_price(K, T);
        ankerl::nanobench::doNotOptimizeAway(fft_result);
    });

    // -----------------------------------------------------------------------
    // 3. ADI PDE solver (Ns=100, Nv=50)
    // -----------------------------------------------------------------------

    ADISolver::Config adi_cfg;
    adi_cfg.Ns = 100; adi_cfg.Nv = 50; adi_cfg.Nt = 200;
    ADISolver solver(p, mkt, adi_cfg);

    double volatile adi_result = 0.0;
    bench.minEpochIterations(5).warmup(2)
         .run("ADI PDE solver (Ns=100, Nv=50, Nt=200)", [&] {
        adi_result = solver.call_price(K, T);
        ankerl::nanobench::doNotOptimizeAway(adi_result);
    });

    // -----------------------------------------------------------------------
    // 4. Full calibration (DE + LM, 30 options)
    // -----------------------------------------------------------------------

    MarketData calib_data = make_calib_data(p, 30);

    Calibrator::Config cal_cfg;
    cal_cfg.max_iter_de   = 200;
    cal_cfg.de_population = 10;
    cal_cfg.max_iter_lm   = 100;
    cal_cfg.n_starts      = 1;
    cal_cfg.verbose       = false;

    double volatile cal_result = 0.0;
    bench.minEpochIterations(1).warmup(0)
         .run("Full calibration DE+LM (30 options)", [&] {
        Calibrator cal(cal_cfg);
        CalibrationResult res = cal.calibrate(calib_data);
        cal_result = res.rmse;
        ankerl::nanobench::doNotOptimizeAway(cal_result);
    });

    // -----------------------------------------------------------------------
    // Summary table
    // -----------------------------------------------------------------------

    std::printf("\n====================================================================\n");
    std::printf("  Method                        | Median      | Note\n");
    std::printf("  ----------------------------- | ----------- | -----------------\n");
    std::printf("  Black-Scholes                 | (see above) | baseline\n");
    std::printf("  Carr-Madan FFT                | (see above) | 4096 strikes/call\n");
    std::printf("  ADI FD solver                 | (see above) | single (K,T)\n");
    std::printf("  Full calibration DE+LM        | (see above) | 30 options, ms\n");
    std::printf("====================================================================\n\n");

    std::printf("Prices computed:\n");
    std::printf("  BS   = %.4f\n", (double)bs_result);
    std::printf("  FFT  = %.4f\n", (double)fft_result);
    std::printf("  ADI  = %.4f\n", (double)adi_result);
    std::printf("  Calibration RMSE = %.2f bps\n", (double)cal_result * 1e4);

    return 0;
}
