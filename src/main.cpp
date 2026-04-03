/**
 * @file main.cpp
 * @brief Heston calibration engine CLI.
 *
 * Usage:
 *   ./heston_demo [data/spx_options_sample.csv]
 *
 * If no file is provided, generates synthetic Heston data with known
 * parameters as ground truth.
 */

#include "heston/calibrator.hpp"
#include "heston/carr_madan.hpp"
#include "heston/characteristic_fn.hpp"
#include "heston/greeks.hpp"
#include "heston/types.hpp"
#include "heston/vol_surface.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace heston;

// ---------------------------------------------------------------------------
// CSV parser
// ---------------------------------------------------------------------------

static MarketData load_csv(const std::string& path, double spot, double rate, double div)
{
    MarketData mkt;
    mkt.spot      = spot;
    mkt.rate      = rate;
    mkt.div_yield = div;

    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "[main] Could not open %s\n", path.c_str());
        return mkt;
    }

    std::string line;
    std::getline(f, line); // skip header

    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        OptionData od{};
        char comma;
        // strike,expiry_years,market_price,market_iv,bid,ask,open_interest,type
        if (!(ss >> od.strike >> comma >> od.expiry >> comma
                 >> od.market_price >> comma >> od.market_iv >> comma
                 >> od.bid >> comma >> od.ask >> comma
                 >> od.open_interest >> comma >> od.type)) {
            continue;
        }
        mkt.options.push_back(od);
    }

    std::printf("[main] Loaded %zu options from %s\n", mkt.options.size(), path.c_str());
    return mkt;
}

// ---------------------------------------------------------------------------
// Synthetic data generator
//
// Generates Heston model prices (via FFT) for known parameters, then inverts
// to BS implied vols.  This gives a "ground truth" calibration target.
// ---------------------------------------------------------------------------

static MarketData generate_synthetic(const HestonParams& true_params)
{
    const double S0 = 100.0, r = 0.05, q = 0.02;

    MarketData mkt;
    mkt.spot      = S0;
    mkt.rate      = r;
    mkt.div_yield = q;

    const std::vector<double> expiries = {0.25, 0.5, 1.0};
    const int n_strikes = 10;
    const double mono_lo = 0.85, mono_hi = 1.15;

    HestonCharacteristicFn cf(true_params, mkt);
    CarrMadanPricer pricer(cf);

    for (double T : expiries) {
        const double F = S0 * std::exp((r - q) * T);
        const double dm = (mono_hi - mono_lo) / (n_strikes - 1);
        for (int i = 0; i < n_strikes; ++i) {
            const double K = F * (mono_lo + dm * static_cast<double>(i));
            const double price = pricer.call_price(K, T);
            const double iv    = CarrMadanPricer::implied_vol(price, S0, K, T, r, q);

            if (!std::isfinite(iv) || iv <= 0.0 || iv > 1.0) continue;

            OptionData od{};
            od.strike        = K;
            od.expiry        = T;
            od.market_price  = price;
            od.market_iv     = iv;
            od.bid           = price * 0.98;
            od.ask           = price * 1.02;
            od.open_interest = 500.0;
            od.type          = 0; // call
            mkt.options.push_back(od);
        }
    }

    std::printf("[main] Generated %zu synthetic options.\n", mkt.options.size());
    std::printf("[main] True parameters:\n");
    true_params.print();

    return mkt;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    std::printf("===================================================\n");
    std::printf("  Heston Stochastic Volatility Calibration Engine  \n");
    std::printf("===================================================\n\n");

    // -----------------------------------------------------------------------
    // 1. Load or generate market data
    // -----------------------------------------------------------------------

    // True Heston parameters (for synthetic data or comparison)
    const HestonParams true_params = {
        .v0    = 0.04,   // σ₀ = 20%
        .kappa = 1.5,
        .theta = 0.04,   // σ_∞ = 20%
        .xi    = 0.3,
        .rho   = -0.7
    };

    MarketData mkt;
    if (argc >= 2) {
        mkt = load_csv(argv[1], 100.0, 0.05, 0.02);
    }

    if (mkt.options.empty()) {
        std::printf("[main] Using synthetic Heston data (no market file provided).\n\n");
        mkt = generate_synthetic(true_params);
    }

    // Filter to liquid options
    MarketData liquid = mkt.filter_liquid();
    if (liquid.options.empty()) {
        std::printf("[main] Warning: no options passed liquidity filter, using all.\n");
        liquid = mkt;
    }
    std::printf("\n[main] Using %zu liquid options for calibration.\n\n",
                liquid.options.size());

    // -----------------------------------------------------------------------
    // 2. Calibrate
    // -----------------------------------------------------------------------

    Calibrator::Config cal_cfg;
    cal_cfg.verbose      = true;
    cal_cfg.max_iter_de  = 500;
    cal_cfg.de_population = 15;
    cal_cfg.max_iter_lm  = 300;
    cal_cfg.n_starts     = 3;

    Calibrator calibrator(cal_cfg);
    CalibrationResult result = calibrator.calibrate(liquid);

    std::printf("\n=== Calibration Complete ===\n");
    result.print();
    Calibrator::validate(result);

    // -----------------------------------------------------------------------
    // 3. Greeks for ATM option at 3M expiry
    // -----------------------------------------------------------------------

    std::printf("\n=== Greeks (ATM, T=0.25) ===\n");
    const double K_atm = mkt.spot;
    const double T_3m  = 0.25;

    GreeksCalculator gc(result.params, liquid);
    Greeks g = gc.pathwise(K_atm, T_3m);
    g.print();

    // -----------------------------------------------------------------------
    // 4. Export vol surface to CSV for Python plotting
    // -----------------------------------------------------------------------

    std::printf("\n=== Building Vol Surface ===\n");

    // Ensure results/ directory exists
    (void)std::system("mkdir -p results");

    VolSurface surface(result.params, liquid);
    surface.export_csv("results/calibration_output.csv", liquid);

    const bool no_arb = surface.check_no_arbitrage();
    std::printf("[main] Static arbitrage check: %s\n", no_arb ? "PASS" : "FAIL");

    // -----------------------------------------------------------------------
    // 5. Summary
    // -----------------------------------------------------------------------

    std::printf("\n=== Summary ===\n");
    std::printf("  Calibrated parameters:\n");
    result.params.print();
    std::printf("  RMSE: %.2f bps | Max error: %.2f bps\n",
                result.rmse * 1e4, result.max_error * 1e4);
    std::printf("  Runtime: %.1f ms\n", result.runtime_ms);
    std::printf("\n  Next steps:\n");
    std::printf("    python3 python/plot_surface.py\n");
    std::printf("    ./build/bench/bench_pricing\n");
    std::printf("    ./build/tests/heston_tests\n");

    return 0;
}
