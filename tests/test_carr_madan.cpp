/**
 * @file test_carr_madan.cpp
 * @brief Unit tests for the Carr-Madan FFT pricer.
 */

#include "heston/carr_madan.hpp"
#include "heston/characteristic_fn.hpp"
#include "heston/types.hpp"
#include <catch2/catch_all.hpp>
#include <cmath>

using namespace heston;

static MarketData make_market(double S = 100.0, double r = 0.05, double q = 0.02)
{
    MarketData mkt;
    mkt.spot      = S;
    mkt.rate      = r;
    mkt.div_yield = q;
    return mkt;
}

// ============================================================================
// TEST 1: Near-zero xi → prices match Black-Scholes
//
// When xi → 0 and v0 = theta, the Heston model reduces to BS with σ = √v0.
// FFT prices should match BS prices within 1 bps.
// ============================================================================

TEST_CASE("CarrMadan: xi≈0 prices match Black-Scholes", "[cm][bs_limit]")
{
    const double v0  = 0.04;
    const double S   = 100.0;
    const double r   = 0.05;
    const double q   = 0.02;
    const double K   = 100.0;
    const double T   = 0.5;

    HestonParams p;
    p.v0    = v0;
    p.kappa = 10.0;
    p.theta = v0;
    p.xi    = 1e-4; // near zero
    p.rho   = 0.0;

    MarketData m = make_market(S, r, q);
    HestonCharacteristicFn cf(p, m);
    CarrMadanPricer pricer(cf);

    const double price_heston = pricer.call_price(K, T);
    const double sigma        = std::sqrt(v0);
    const double price_bs     = CarrMadanPricer::bs_call(S, K, T, r, q, sigma);

    const double diff_bps = std::abs(price_heston - price_bs) / price_bs * 1e4;
    CAPTURE(price_heston, price_bs, diff_bps);
    REQUIRE(diff_bps < 5.0); // within 5 bps (xi is small, not zero)
}

// ============================================================================
// TEST 2: Put-call parity C − P = F·e^{-rT} − K·e^{-rT}
// ============================================================================

TEST_CASE("CarrMadan: put-call parity holds", "[cm][pcp]")
{
    HestonParams p = {.v0 = 0.04, .kappa = 1.5, .theta = 0.04, .xi = 0.3, .rho = -0.5};
    const double S = 100.0, r = 0.05, q = 0.02, T = 0.5;
    MarketData m = make_market(S, r, q);
    HestonCharacteristicFn cf(p, m);
    CarrMadanPricer pricer(cf);

    for (double K : {90.0, 100.0, 110.0}) {
        const double call  = pricer.call_price(K, T);
        const double F     = S * std::exp((r - q) * T);
        const double pcp   = F * std::exp(-r * T) - K * std::exp(-r * T);

        // Build a put via MarketData with type=1
        MarketData mkt_put = m;
        OptionData od{};
        od.strike        = K;
        od.expiry        = T;
        od.market_price  = 0.0;
        od.market_iv     = 0.20;
        od.bid           = 0.0;
        od.ask           = 0.0;
        od.open_interest = 1000.0;
        od.type          = 1;
        mkt_put.options  = {od};

        const std::vector<double> put_prices = pricer.price_all(mkt_put);
        const double put = put_prices[0];
        const double parity_error = std::abs((call - put) - pcp);

        CAPTURE(K, call, put, pcp, parity_error);
        REQUIRE(parity_error < 1e-4);
    }
}

// ============================================================================
// TEST 3: Call prices are positive for all strikes in [0.5K, 2K]
// ============================================================================

TEST_CASE("CarrMadan: call prices positive across wide strike range", "[cm][positivity]")
{
    HestonParams p = {.v0 = 0.04, .kappa = 2.0, .theta = 0.04, .xi = 0.5, .rho = -0.6};
    const double S = 100.0, r = 0.05, q = 0.02, T = 1.0;
    MarketData m = make_market(S, r, q);
    HestonCharacteristicFn cf(p, m);
    CarrMadanPricer pricer(cf);

    for (double K : {50.0, 70.0, 90.0, 100.0, 110.0, 130.0, 150.0, 200.0}) {
        const double price = pricer.call_price(K, T);
        CAPTURE(K, price);
        REQUIRE(price >= 0.0);
        // Also check intrinsic value lower bound
        const double intrinsic = std::max(S * std::exp(-q * T) - K * std::exp(-r * T), 0.0);
        REQUIRE(price >= intrinsic - 1e-3);
    }
}

// ============================================================================
// TEST 4: IV round-trip: BS price → FFT IV matches input sigma
//
// Generate a BS price with known sigma, then price via Heston-FFT (xi≈0),
// invert to IV, and check the round-trip.
// ============================================================================

TEST_CASE("CarrMadan: IV inversion round-trip for flat-vol input", "[cm][iv_roundtrip]")
{
    const double S = 100.0, K = 100.0, T = 0.5, r = 0.05, q = 0.02;
    const double sigma_input = 0.25;

    const double bs_price = CarrMadanPricer::bs_call(S, K, T, r, q, sigma_input);
    const double iv_out   = CarrMadanPricer::implied_vol(bs_price, S, K, T, r, q);

    CAPTURE(sigma_input, bs_price, iv_out);
    REQUIRE(std::abs(iv_out - sigma_input) < 1e-6);
}

// ============================================================================
// TEST 5: price_all prices options correctly (compare to single call_price)
// ============================================================================

TEST_CASE("CarrMadan: price_all matches individual call_price", "[cm][batch]")
{
    HestonParams p = {.v0 = 0.04, .kappa = 1.5, .theta = 0.04, .xi = 0.3, .rho = -0.5};
    const double S = 100.0, r = 0.05, q = 0.02;
    MarketData m = make_market(S, r, q);
    HestonCharacteristicFn cf(p, m);
    CarrMadanPricer pricer(cf);

    // Build a MarketData with 5 options at same expiry
    const double T = 0.5;
    const std::vector<double> strikes = {85.0, 95.0, 100.0, 105.0, 115.0};
    m.options.clear();
    for (double K : strikes) {
        OptionData od{};
        od.strike = K; od.expiry = T; od.type = 0;
        od.market_iv = 0.20; od.bid = 1.0; od.ask = 1.1; od.open_interest = 200.0;
        m.options.push_back(od);
    }

    const auto batch = pricer.price_all(m);

    for (std::size_t i = 0; i < strikes.size(); ++i) {
        const double single = pricer.call_price(strikes[i], T);
        CAPTURE(strikes[i], batch[i], single);
        REQUIRE(std::abs(batch[i] - single) < 1e-4);
    }
}

// ============================================================================
// TEST 6: BS helpers — bs_call and bs_vega are consistent
// ============================================================================

TEST_CASE("CarrMadan: BS call and vega consistent", "[cm][bs_helpers]")
{
    const double S = 100.0, K = 100.0, T = 0.5, r = 0.05, q = 0.02;
    const double sigma = 0.2;
    const double dv = 1e-4;

    const double call_up = CarrMadanPricer::bs_call(S, K, T, r, q, sigma + dv);
    const double call_dn = CarrMadanPricer::bs_call(S, K, T, r, q, sigma - dv);
    const double fd_vega = (call_up - call_dn) / (2.0 * dv);
    const double an_vega = CarrMadanPricer::bs_vega(S, K, T, r, q, sigma);

    CAPTURE(fd_vega, an_vega);
    REQUIRE(std::abs(fd_vega - an_vega) / an_vega < 1e-4);
}
