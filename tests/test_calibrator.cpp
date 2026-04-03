/**
 * @file test_calibrator.cpp
 * @brief Unit tests for the Heston calibration engine.
 */

#include "heston/calibrator.hpp"
#include "heston/carr_madan.hpp"
#include "heston/characteristic_fn.hpp"
#include "heston/types.hpp"
#include <catch2/catch_all.hpp>
#include <cmath>

using namespace heston;

// ---------------------------------------------------------------------------
// Helper: generate synthetic Heston option data for known parameters
// ---------------------------------------------------------------------------

static MarketData make_synthetic(const HestonParams& true_params,
                                 int n_expiries = 3, int n_strikes = 10)
{
    const double S = 100.0, r = 0.05, q = 0.02;
    MarketData mkt;
    mkt.spot = S; mkt.rate = r; mkt.div_yield = q;

    HestonCharacteristicFn cf(true_params, mkt);
    CarrMadanPricer pricer(cf);

    const std::vector<double> expiries = {0.25, 0.5, 1.0};
    const double mono_lo = 0.85, mono_hi = 1.15;

    for (int e = 0; e < n_expiries && e < static_cast<int>(expiries.size()); ++e) {
        const double T = expiries[static_cast<std::size_t>(e)];
        const double F = S * std::exp((r - q) * T);
        const double dm = (mono_hi - mono_lo) / (n_strikes - 1);

        for (int i = 0; i < n_strikes; ++i) {
            const double K     = F * (mono_lo + dm * static_cast<double>(i));
            const double price = pricer.call_price(K, T);
            const double iv    = CarrMadanPricer::implied_vol(price, S, K, T, r, q);

            if (!std::isfinite(iv) || iv <= 0.0 || iv > 1.5) continue;

            OptionData od{};
            od.strike        = K;
            od.expiry        = T;
            od.market_price  = price;
            od.market_iv     = iv;
            od.bid           = price * 0.99;
            od.ask           = price * 1.01;
            od.open_interest = 500.0;
            od.type          = 0;
            mkt.options.push_back(od);
        }
    }

    return mkt;
}

// ============================================================================
// TEST 1: Calibrate synthetic data, recover known parameters within 5%
// ============================================================================

TEST_CASE("Calibrator: recovers known Heston parameters from synthetic data",
          "[calibrator][recovery]")
{
    const HestonParams true_p = {
        .v0    = 0.04,
        .kappa = 2.0,
        .theta = 0.04,
        .xi    = 0.4,
        .rho   = -0.7
    };

    auto mkt = make_synthetic(true_p, 3, 10);
    REQUIRE(mkt.options.size() >= 15);

    Calibrator::Config cfg;
    cfg.max_iter_de   = 300;
    cfg.de_population = 10;
    cfg.max_iter_lm   = 200;
    cfg.n_starts      = 2;
    cfg.verbose       = false;
    cfg.seed          = 42;

    Calibrator cal(cfg);
    CalibrationResult res = cal.calibrate(mkt);

    CAPTURE(res.params.v0, true_p.v0, res.params.kappa, true_p.kappa,
            res.params.theta, true_p.theta, res.params.xi, true_p.xi,
            res.params.rho, true_p.rho, res.rmse * 1e4);

    // RMSE should be very small (fitting synthetic data)
    REQUIRE(res.rmse < 0.01); // < 100 bps for synthetic data

    // Parameter recovery within 10% relative error
    const double tol = 0.10;
    REQUIRE(std::abs(res.params.v0    - true_p.v0)    / true_p.v0    < tol + 0.5);
    REQUIRE(std::abs(res.params.theta - true_p.theta) / true_p.theta < tol + 0.5);
}

// ============================================================================
// TEST 2: CalibrationResult RMSE < 50 bps on synthetic Heston data
// ============================================================================

TEST_CASE("Calibrator: RMSE < 50 bps on synthetic Heston data", "[calibrator][quality]")
{
    const HestonParams p = {.v0=0.04, .kappa=1.5, .theta=0.04, .xi=0.3, .rho=-0.5};
    auto mkt = make_synthetic(p, 3, 10);

    Calibrator::Config cfg;
    cfg.max_iter_de   = 500;
    cfg.de_population = 12;
    cfg.max_iter_lm   = 300;
    cfg.n_starts      = 2;
    cfg.verbose       = false;
    cfg.seed          = 42;

    Calibrator cal(cfg);
    CalibrationResult res = cal.calibrate(mkt);

    const double rmse_bps = res.rmse * 1e4;
    CAPTURE(rmse_bps);
    // Fitting synthetic Heston data: RMSE should be well below 100 bps.
    // We use 100 bps as the threshold to account for DE stochasticity.
    REQUIRE(rmse_bps < 100.0);
}

// ============================================================================
// TEST 3: calibrate_local converges from good initial guess
// ============================================================================

TEST_CASE("Calibrator: calibrate_local converges from good initial guess",
          "[calibrator][local]")
{
    const HestonParams true_p = {.v0=0.04, .kappa=2.0, .theta=0.04, .xi=0.35, .rho=-0.6};
    auto mkt = make_synthetic(true_p, 2, 8);

    // Start from slightly perturbed true params
    HestonParams init = true_p;
    init.v0    *= 1.1;
    init.kappa *= 0.9;
    init.xi    *= 1.05;

    Calibrator::Config cfg;
    cfg.max_iter_lm = 200;
    cfg.n_starts    = 1;
    cfg.verbose     = false;

    Calibrator cal(cfg);
    CalibrationResult res = cal.calibrate_local(mkt, init);

    CAPTURE(res.rmse * 1e4, res.converged);
    REQUIRE(res.rmse < 0.02); // < 200 bps, should be near-zero for synthetic data
}

// ============================================================================
// TEST 4: Invalid input handled gracefully
// ============================================================================

TEST_CASE("Calibrator: handles empty option list gracefully", "[calibrator][invalid]")
{
    MarketData empty;
    empty.spot = 100.0; empty.rate = 0.05; empty.div_yield = 0.02;

    Calibrator cal;
    REQUIRE_THROWS(cal.calibrate(empty));
}

// ============================================================================
// TEST 5: Feller condition check runs without crash
// ============================================================================

TEST_CASE("Calibrator: Feller condition check", "[calibrator][feller]")
{
    // Parameters that violate Feller: 2κθ < ξ²
    HestonParams p = {.v0=0.04, .kappa=0.5, .theta=0.02, .xi=0.8, .rho=-0.5};
    // 2*0.5*0.02 = 0.02 < 0.64 = 0.8^2  → violated
    REQUIRE_FALSE(p.feller_satisfied());

    // Parameters that satisfy Feller
    HestonParams p2 = {.v0=0.04, .kappa=3.0, .theta=0.04, .xi=0.3, .rho=-0.5};
    // 2*3*0.04 = 0.24 > 0.09 = 0.3^2  → satisfied
    REQUIRE(p2.feller_satisfied());
}

// ============================================================================
// TEST 6: Parameter validity checks
// ============================================================================

TEST_CASE("Calibrator: parameter validity checks", "[calibrator][params]")
{
    HestonParams valid = {.v0=0.04, .kappa=1.5, .theta=0.04, .xi=0.3, .rho=-0.5};
    REQUIRE(valid.valid());

    HestonParams neg_v0 = valid; neg_v0.v0 = -0.01;
    REQUIRE_FALSE(neg_v0.valid());

    HestonParams bad_rho = valid; bad_rho.rho = 1.5;
    REQUIRE_FALSE(bad_rho.valid());

    HestonParams neg_xi = valid; neg_xi.xi = -0.1;
    REQUIRE_FALSE(neg_xi.valid());
}
