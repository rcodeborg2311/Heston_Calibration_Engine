/**
 * @file vol_surface.cpp
 * @brief Implied-volatility surface implementation.
 */

#include "heston/vol_surface.hpp"
#include "heston/carr_madan.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace heston {

// ---------------------------------------------------------------------------
// Constructor: evaluate full surface grid
// ---------------------------------------------------------------------------

VolSurface::VolSurface(const HestonParams& params, const MarketData& mkt)
    : VolSurface(params, mkt, Config{})
{}

VolSurface::VolSurface(const HestonParams& params, const MarketData& mkt, Config cfg)
    : params_(params), mkt_(mkt), cfg_(cfg)
{
    build();
}

// ---------------------------------------------------------------------------
// Build the vol surface on the configured grid
// ---------------------------------------------------------------------------

void VolSurface::build()
{
    const int    nK  = cfg_.n_strikes;
    const int    nT  = cfg_.n_expiries;
    const double S   = mkt_.spot;
    const double r   = mkt_.rate;
    const double q   = mkt_.div_yield;

    K_grid_.resize(static_cast<std::size_t>(nK));
    T_grid_.resize(static_cast<std::size_t>(nT));
    iv_.assign(static_cast<std::size_t>(nK),
               std::vector<double>(static_cast<std::size_t>(nT), 0.0));

    // Uniform moneyness grid
    const double dm = (cfg_.moneyness_hi - cfg_.moneyness_lo) / (nK - 1);
    for (int i = 0; i < nK; ++i) {
        K_grid_[static_cast<std::size_t>(i)] = S * (cfg_.moneyness_lo + dm * static_cast<double>(i));
    }

    // Log-uniform expiry grid
    const double logT_lo = std::log(cfg_.expiry_lo);
    const double logT_hi = std::log(cfg_.expiry_hi);
    const double dlogT   = (logT_hi - logT_lo) / (nT - 1);
    for (int j = 0; j < nT; ++j) {
        T_grid_[static_cast<std::size_t>(j)] = std::exp(logT_lo + dlogT * static_cast<double>(j));
    }

    // Create characteristic function once (model params fixed)
    HestonCharacteristicFn cf(params_, mkt_);
    CarrMadanPricer pricer(cf);

    // For each expiry, price all strikes via one FFT call
    for (int j = 0; j < nT; ++j) {
        const double T = T_grid_[static_cast<std::size_t>(j)];
        const double F = S * std::exp((r - q) * T);

        // Build mini-MarketData with all strikes at this expiry
        MarketData slice;
        slice.spot      = S;
        slice.rate      = r;
        slice.div_yield = q;
        slice.options.reserve(static_cast<std::size_t>(nK));

        for (int i = 0; i < nK; ++i) {
            OptionData od{};
            od.strike       = K_grid_[static_cast<std::size_t>(i)];
            od.expiry       = T;
            od.market_price = 0.0;
            od.market_iv    = 0.0;
            od.bid          = 0.0;
            od.ask          = 0.0;
            od.open_interest = 0.0;
            od.type         = 0; // call
            slice.options.push_back(od);
        }

        const std::vector<double> ivs = pricer.implied_vols(slice);

        for (int i = 0; i < nK; ++i) {
            iv_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                std::isfinite(ivs[static_cast<std::size_t>(i)]) ? ivs[static_cast<std::size_t>(i)] : 0.0;
        }

        (void)F;
    }
}

// ---------------------------------------------------------------------------
// Query: bilinear interpolation
// ---------------------------------------------------------------------------

double VolSurface::implied_vol(double K, double T) const
{
    // Find K index
    const auto iK_it = std::lower_bound(K_grid_.begin(), K_grid_.end(), K);
    std::size_t iK = static_cast<std::size_t>(iK_it - K_grid_.begin());
    if (iK == 0) iK = 1;
    if (iK >= K_grid_.size()) iK = K_grid_.size() - 1;
    const std::size_t iK0 = iK - 1;

    // Find T index
    const auto iT_it = std::lower_bound(T_grid_.begin(), T_grid_.end(), T);
    std::size_t iT = static_cast<std::size_t>(iT_it - T_grid_.begin());
    if (iT == 0) iT = 1;
    if (iT >= T_grid_.size()) iT = T_grid_.size() - 1;
    const std::size_t iT0 = iT - 1;

    const double wK = (K - K_grid_[iK0]) / (K_grid_[iK] - K_grid_[iK0]);
    const double wT = (T - T_grid_[iT0]) / (T_grid_[iT] - T_grid_[iT0]);

    return (1.0 - wK) * (1.0 - wT) * iv_[iK0][iT0]
         + wK         * (1.0 - wT) * iv_[iK ][iT0]
         + (1.0 - wK) * wT         * iv_[iK0][iT ]
         + wK         * wT         * iv_[iK ][iT ];
}

// ---------------------------------------------------------------------------
// CSV export
// ---------------------------------------------------------------------------

void VolSurface::export_csv(const std::string& path, const MarketData& market) const
{
    std::ofstream out(path);
    if (!out) throw std::runtime_error("VolSurface::export_csv: cannot open " + path);

    out << "strike,expiry,market_iv,model_iv\n";

    const double S = mkt_.spot;

    // First, write market observations with corresponding model IVs
    for (const auto& opt : market.options) {
        const double model = implied_vol(opt.strike, opt.expiry);
        out << opt.strike << "," << opt.expiry << ","
            << opt.market_iv << "," << model << "\n";
    }

    // Then write the dense model grid (market_iv = NaN for grid points)
    for (std::size_t j = 0; j < T_grid_.size(); ++j) {
        for (std::size_t i = 0; i < K_grid_.size(); ++i) {
            const double K = K_grid_[i];
            const double T = T_grid_[j];
            // Skip if outside liquid moneyness range
            if (K / S < 0.5 || K / S > 1.5) continue;
            out << K << "," << T << ",,"  // empty market_iv
                << iv_[i][j] << "\n";
        }
    }

    std::printf("[VolSurface] Exported to %s\n", path.c_str());
}

// ---------------------------------------------------------------------------
// No-arbitrage check
// ---------------------------------------------------------------------------

bool VolSurface::check_no_arbitrage() const
{
    bool ok = true;
    const int nK = static_cast<int>(K_grid_.size());
    const int nT = static_cast<int>(T_grid_.size());

    // Calendar spread: total variance σ²T must be non-decreasing in T
    for (int i = 0; i < nK; ++i) {
        for (int j = 0; j < nT - 1; ++j) {
            const double tv1 = iv_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] *
                               iv_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] *
                               T_grid_[static_cast<std::size_t>(j)];
            const double tv2 = iv_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j + 1)] *
                               iv_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j + 1)] *
                               T_grid_[static_cast<std::size_t>(j + 1)];
            if (tv2 < tv1 - 1e-6) {
                std::fprintf(stderr,
                    "[VolSurface] Calendar arbitrage at K=%.2f T=%.3f→%.3f: "
                    "σ²T drops from %.4f to %.4f\n",
                    K_grid_[static_cast<std::size_t>(i)],
                    T_grid_[static_cast<std::size_t>(j)],
                    T_grid_[static_cast<std::size_t>(j + 1)],
                    tv1, tv2);
                ok = false;
            }
        }
    }

    return ok;
}

} // namespace heston
