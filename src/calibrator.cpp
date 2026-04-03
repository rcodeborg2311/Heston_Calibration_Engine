/**
 * @file calibrator.cpp
 * @brief Heston calibration engine: Differential Evolution + Levenberg-Marquardt.
 *
 * Optimization strategy:
 *
 * Phase 1 — Differential Evolution (DE/rand/1/bin, Storn & Price 1997):
 *   Global stochastic search that avoids local minima common in the
 *   5-parameter Heston surface.  Runs max_iter_de generations on a
 *   population of de_population parameter vectors.
 *
 * Phase 2 — Levenberg-Marquardt (LM):
 *   Gradient-based refinement from the DE solution.  Jacobian is computed
 *   via central finite differences.  The 5×5 normal equations are solved
 *   directly (Gaussian elimination).
 *
 * Objective: weighted RMSE of Black-Scholes implied volatilities.
 */

#include "heston/calibrator.hpp"
#include "heston/carr_madan.hpp"
#include "heston/characteristic_fn.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace heston {

// ---------------------------------------------------------------------------
// Parameter encoding helpers
// ---------------------------------------------------------------------------

HestonParams Calibrator::to_params(const ParamVec& x) noexcept
{
    return {x[0], x[1], x[2], x[3], x[4]};
}

Calibrator::ParamVec Calibrator::to_vec(const HestonParams& p) noexcept
{
    return {p.v0, p.kappa, p.theta, p.xi, p.rho};
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

Calibrator::Calibrator()
    : Calibrator(Config{})
{}

Calibrator::Calibrator(Config cfg)
    : cfg_(cfg), rng_(cfg.seed)
{}

// ---------------------------------------------------------------------------
// Feasibility projection
// ---------------------------------------------------------------------------

HestonParams Calibrator::project_constraints(const HestonParams& p) noexcept
{
    return {
        std::clamp(p.v0,    V0_MIN, V0_MAX),
        std::clamp(p.kappa, K_MIN,  K_MAX),
        std::clamp(p.theta, TH_MIN, TH_MAX),
        std::clamp(p.xi,    XI_MIN, XI_MAX),
        std::clamp(p.rho,   RH_MIN, RH_MAX)
    };
}

void Calibrator::check_feller(const HestonParams& p) noexcept
{
    if (!p.feller_satisfied()) {
        std::fprintf(stderr,
            "[WARNING] Feller condition violated: 2κθ=%.4f < ξ²=%.4f. "
            "Variance may reach zero.\n",
            2.0 * p.kappa * p.theta, p.xi * p.xi);
    }
}

// ---------------------------------------------------------------------------
// Random parameter sample (uniform on feasible box)
// ---------------------------------------------------------------------------

HestonParams Calibrator::random_params()
{
    std::uniform_real_distribution<double> u_v0   (V0_MIN, V0_MAX);
    std::uniform_real_distribution<double> u_kappa(K_MIN,  K_MAX);
    std::uniform_real_distribution<double> u_theta(TH_MIN, TH_MAX);
    std::uniform_real_distribution<double> u_xi   (XI_MIN, XI_MAX);
    std::uniform_real_distribution<double> u_rho  (RH_MIN, RH_MAX);

    return {u_v0(rng_), u_kappa(rng_), u_theta(rng_), u_xi(rng_), u_rho(rng_)};
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

std::vector<double> Calibrator::compute_weights(const MarketData& mkt) const
{
    const std::size_t n = mkt.options.size();
    std::vector<double> w(n, 1.0);

    if (!cfg_.use_weights) return w;

    for (std::size_t i = 0; i < n; ++i) {
        const double sp = mkt.options[i].spread();
        w[i] = (sp > 0.0) ? 1.0 / sp : 1.0;
    }

    // Normalise so Σw = n
    const double total = std::accumulate(w.begin(), w.end(), 0.0);
    if (total > 0.0) {
        const double scale = static_cast<double>(n) / total;
        for (auto& wi : w) wi *= scale;
    }

    return w;
}

// ---------------------------------------------------------------------------
// Residuals and loss
// ---------------------------------------------------------------------------

std::vector<double> Calibrator::residuals(const HestonParams& p,
                                           const MarketData& mkt) const
{
    const std::size_t n = mkt.options.size();
    const std::vector<double> weights = compute_weights(mkt);

    std::vector<double> res(n, 0.0);

    if (!p.valid()) {
        std::fill(res.begin(), res.end(), 1e3);
        return res;
    }

    try {
        HestonCharacteristicFn cf(p, mkt);
        CarrMadanPricer pricer(cf);
        const std::vector<double> model_ivs = pricer.implied_vols(mkt);

        for (std::size_t i = 0; i < n; ++i) {
            const double sigma_m = mkt.options[i].market_iv;
            const double sigma_h = std::isfinite(model_ivs[i]) ? model_ivs[i] : 0.5;
            res[i] = std::sqrt(weights[i]) * (sigma_h - sigma_m);
        }
    } catch (...) {
        std::fill(res.begin(), res.end(), 1e3);
    }

    return res;
}

double Calibrator::evaluate_loss(const HestonParams& p, const MarketData& mkt) const
{
    const auto res = residuals(p, mkt);
    const double ssq = std::inner_product(res.begin(), res.end(), res.begin(), 0.0);
    return std::sqrt(ssq / static_cast<double>(res.size()));
}

// ---------------------------------------------------------------------------
// Differential Evolution
// ---------------------------------------------------------------------------

HestonParams Calibrator::differential_evolution(const MarketData& mkt)
{
    const int    NP  = cfg_.de_population;
    const int    D   = 5; // dimensions
    const double F   = cfg_.de_F;
    const double CR  = cfg_.de_CR;
    const int    Gmax = cfg_.max_iter_de;

    // Initialize population
    std::vector<ParamVec> pop(static_cast<std::size_t>(NP));
    std::vector<double>   fitness(static_cast<std::size_t>(NP));

    for (int k = 0; k < NP; ++k) {
        const HestonParams rp = random_params();
        pop[static_cast<std::size_t>(k)]     = to_vec(rp);
        fitness[static_cast<std::size_t>(k)] = evaluate_loss(rp, mkt);
    }

    std::uniform_int_distribution<int> u_idx(0, NP - 1);
    std::uniform_int_distribution<int> u_dim(0, D - 1);
    std::uniform_real_distribution<double> u_01(0.0, 1.0);

    const ParamVec lo = {V0_MIN, K_MIN,  TH_MIN, XI_MIN, RH_MIN};
    const ParamVec hi = {V0_MAX, K_MAX,  TH_MAX, XI_MAX, RH_MAX};

    double best_fitness = *std::min_element(fitness.begin(), fitness.end());
    int    no_improve   = 0;

    for (int gen = 0; gen < Gmax; ++gen) {
        for (int i = 0; i < NP; ++i) {
            // Pick three distinct random indices ≠ i
            int r1, r2, r3;
            do { r1 = u_idx(rng_); } while (r1 == i);
            do { r2 = u_idx(rng_); } while (r2 == i || r2 == r1);
            do { r3 = u_idx(rng_); } while (r3 == i || r3 == r1 || r3 == r2);

            const ParamVec& x  = pop[static_cast<std::size_t>(i)];
            const ParamVec& xr1 = pop[static_cast<std::size_t>(r1)];
            const ParamVec& xr2 = pop[static_cast<std::size_t>(r2)];
            const ParamVec& xr3 = pop[static_cast<std::size_t>(r3)];

            // Mutation: v = xr1 + F * (xr2 - xr3)
            ParamVec v;
            for (int d = 0; d < D; ++d) {
                v[static_cast<std::size_t>(d)] = xr1[static_cast<std::size_t>(d)]
                    + F * (xr2[static_cast<std::size_t>(d)] - xr3[static_cast<std::size_t>(d)]);
                // Clamp to bounds
                v[static_cast<std::size_t>(d)] = std::clamp(
                    v[static_cast<std::size_t>(d)],
                    lo[static_cast<std::size_t>(d)],
                    hi[static_cast<std::size_t>(d)]);
            }

            // Crossover: u_j = v_j if U(0,1)<CR, else x_j
            int j_rand = u_dim(rng_);
            ParamVec trial = x;
            for (int d = 0; d < D; ++d) {
                if (u_01(rng_) < CR || d == j_rand) {
                    trial[static_cast<std::size_t>(d)] = v[static_cast<std::size_t>(d)];
                }
            }

            // Selection
            const double f_trial = evaluate_loss(to_params(trial), mkt);
            if (f_trial <= fitness[static_cast<std::size_t>(i)]) {
                pop[static_cast<std::size_t>(i)]     = trial;
                fitness[static_cast<std::size_t>(i)] = f_trial;
            }
        }

        const double new_best = *std::min_element(fitness.begin(), fitness.end());
        if (new_best < best_fitness - 1e-8) {
            best_fitness = new_best;
            no_improve   = 0;
        } else {
            ++no_improve;
        }

        if (cfg_.verbose && gen % 100 == 0) {
            std::printf("[DE] gen=%d  best_loss=%.6f\n", gen, best_fitness);
        }

        if (no_improve > 200) break; // Early stopping
    }

    // Return best member
    const std::size_t best_idx = static_cast<std::size_t>(
        std::min_element(fitness.begin(), fitness.end()) - fitness.begin());
    return to_params(pop[best_idx]);
}

// ---------------------------------------------------------------------------
// Levenberg-Marquardt step
//
// Normal equations: (J^T W J + λ I) δ = J^T W r
// where W = diag(weights), J is the Jacobian of residuals w.r.t. params.
//
// Central finite differences for Jacobian:
//   J[i][k] = (r_i(p + h·e_k) - r_i(p - h·e_k)) / (2h)
// ---------------------------------------------------------------------------

HestonParams Calibrator::lm_step(const HestonParams& current,
                                   const MarketData& mkt,
                                   double lambda_lm,
                                   double& current_loss) const
{
    constexpr int    D = 5;
    constexpr double h = 1e-5;

    const std::vector<double> r0 = residuals(current, mkt);
    const std::size_t N = r0.size();

    // Jacobian: N x D matrix (stored row-major)
    std::vector<std::vector<double>> J(N, std::vector<double>(D, 0.0));

    const ParamVec p0 = to_vec(current);
    const double   steps[D] = {h, h, h, h, h};

    const ParamVec lo = {V0_MIN, K_MIN,  TH_MIN, XI_MIN, RH_MIN};
    const ParamVec hi = {V0_MAX, K_MAX,  TH_MAX, XI_MAX, RH_MAX};

    for (int k = 0; k < D; ++k) {
        ParamVec pp = p0, pm = p0;
        pp[static_cast<std::size_t>(k)] = std::clamp(p0[static_cast<std::size_t>(k)] + steps[k],
                                                       lo[static_cast<std::size_t>(k)],
                                                       hi[static_cast<std::size_t>(k)]);
        pm[static_cast<std::size_t>(k)] = std::clamp(p0[static_cast<std::size_t>(k)] - steps[k],
                                                       lo[static_cast<std::size_t>(k)],
                                                       hi[static_cast<std::size_t>(k)]);

        const std::vector<double> rp = residuals(to_params(pp), mkt);
        const std::vector<double> rm = residuals(to_params(pm), mkt);
        const double dh = pp[static_cast<std::size_t>(k)] - pm[static_cast<std::size_t>(k)];

        for (std::size_t i = 0; i < N; ++i) {
            J[i][static_cast<std::size_t>(k)] = (rp[i] - rm[i]) / dh;
        }
    }

    // Build J^T J  (5×5 symmetric matrix)
    double JtJ[D][D] = {};
    double Jtr[D]    = {};
    for (std::size_t i = 0; i < N; ++i) {
        for (int k = 0; k < D; ++k) {
            Jtr[k] += J[i][static_cast<std::size_t>(k)] * r0[i];
            for (int l = 0; l < D; ++l) {
                JtJ[k][l] += J[i][static_cast<std::size_t>(k)] * J[i][static_cast<std::size_t>(l)];
            }
        }
    }

    // Add Marquardt damping: (J^T J + λ I) δ = -J^T r
    for (int k = 0; k < D; ++k) {
        JtJ[k][k] += lambda_lm;
        Jtr[k] = -Jtr[k]; // negate for descent
    }

    // Solve 5×5 system via Gaussian elimination with partial pivoting
    double A[D][D+1];
    for (int k = 0; k < D; ++k) {
        for (int l = 0; l < D; ++l) A[k][l] = JtJ[k][l];
        A[k][D] = Jtr[k];
    }

    for (int col = 0; col < D; ++col) {
        // Partial pivot
        int pivot = col;
        for (int row = col + 1; row < D; ++row) {
            if (std::abs(A[row][col]) > std::abs(A[pivot][col])) pivot = row;
        }
        if (pivot != col) {
            for (int l = 0; l <= D; ++l) std::swap(A[col][l], A[pivot][l]);
        }

        if (std::abs(A[col][col]) < 1e-14) continue;

        for (int row = 0; row < D; ++row) {
            if (row == col) continue;
            const double factor = A[row][col] / A[col][col];
            for (int l = col; l <= D; ++l) A[row][l] -= factor * A[col][l];
        }
    }

    // Extract solution δ
    ParamVec delta = {};
    for (int k = 0; k < D; ++k) {
        if (std::abs(A[k][k]) > 1e-14) {
            delta[static_cast<std::size_t>(k)] = A[k][D] / A[k][k];
        }
    }

    // Candidate update
    ParamVec pnew = p0;
    for (int k = 0; k < D; ++k) {
        pnew[static_cast<std::size_t>(k)] += delta[static_cast<std::size_t>(k)];
    }

    HestonParams candidate = project_constraints(to_params(pnew));
    const double new_loss  = evaluate_loss(candidate, mkt);

    if (new_loss < current_loss) {
        current_loss = new_loss;
        return candidate;
    }
    return current; // Reject step
}

// ---------------------------------------------------------------------------
// Two-phase calibration
// ---------------------------------------------------------------------------

CalibrationResult Calibrator::calibrate(const MarketData& mkt)
{
    if (mkt.options.empty()) {
        throw std::invalid_argument("Calibrator::calibrate: empty option list");
    }

    const auto t_start = std::chrono::high_resolution_clock::now();

    // Phase 1: Global DE search
    if (cfg_.verbose) std::printf("[Calibrator] Phase 1: Differential Evolution...\n");
    HestonParams best = differential_evolution(mkt);
    check_feller(best);

    // Phase 2: LM refinement from DE best
    if (cfg_.verbose) std::printf("[Calibrator] Phase 2: Levenberg-Marquardt...\n");
    CalibrationResult res = calibrate_local(mkt, best);

    const auto t_end = std::chrono::high_resolution_clock::now();
    res.runtime_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return res;
}

CalibrationResult Calibrator::calibrate_local(const MarketData& mkt,
                                               const HestonParams& initial_guess)
{
    const auto t_start = std::chrono::high_resolution_clock::now();

    HestonParams current = project_constraints(initial_guess);
    double       loss    = evaluate_loss(current, mkt);
    double       lambda  = 1e-3; // Marquardt damping — start moderate
    int          iter    = 0;
    bool         converged = false;

    for (int it = 0; it < cfg_.max_iter_lm; ++it) {
        const double prev_loss = loss;
        current = lm_step(current, mkt, lambda, loss);
        ++iter;

        if (std::abs(prev_loss - loss) < cfg_.tol * (1.0 + prev_loss)) {
            converged = true;
            break;
        }

        // Adaptive damping: reduce λ on progress, increase on stall
        if (loss < prev_loss) {
            lambda = std::max(lambda / 10.0, 1e-12);
        } else {
            lambda = std::min(lambda * 10.0, 1e6);
        }

        if (cfg_.verbose && it % 50 == 0) {
            std::printf("[LM] iter=%d  loss=%.6f  λ=%.2e\n", it, loss, lambda);
        }
    }

    // Multi-start: try n_starts random restarts, keep best
    for (int s = 1; s < cfg_.n_starts; ++s) {
        HestonParams trial = random_params();
        double trial_loss  = evaluate_loss(trial, mkt);
        double trial_lambda = 1e-3;

        for (int it = 0; it < cfg_.max_iter_lm / cfg_.n_starts; ++it) {
            trial = lm_step(trial, mkt, trial_lambda, trial_loss);
            if (trial_loss < loss) {
                loss    = trial_loss;
                current = trial;
            }
        }
    }

    // Build final result
    CalibrationResult res;
    res.params = current;
    res.n_iterations = iter;
    res.converged    = converged;

    try {
        HestonCharacteristicFn cf(current, mkt);
        CarrMadanPricer pricer(cf);
        res.model_ivs = pricer.implied_vols(mkt);
    } catch (...) {
        res.model_ivs.assign(mkt.options.size(), 0.0);
    }

    res.market_ivs.reserve(mkt.options.size());
    double ssq = 0.0, max_err = 0.0;
    for (std::size_t i = 0; i < mkt.options.size(); ++i) {
        const double sigma_m = mkt.options[i].market_iv;
        const double sigma_h = std::isfinite(res.model_ivs[i]) ? res.model_ivs[i] : 0.0;
        res.market_ivs.push_back(sigma_m);
        const double err = std::abs(sigma_h - sigma_m);
        ssq += err * err;
        max_err = std::max(max_err, err);
    }

    res.rmse      = std::sqrt(ssq / static_cast<double>(mkt.options.size()));
    res.max_error = max_err;

    const auto t_end = std::chrono::high_resolution_clock::now();
    res.runtime_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return res;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

void Calibrator::validate(const CalibrationResult& result)
{
    std::printf("\n=== Calibration Validation ===\n");
    result.print();

    const double rmse_bps = result.rmse * 1e4;
    const double max_bps  = result.max_error * 1e4;

    std::printf("Quality checks:\n");
    std::printf("  RMSE < 50 bps:     %s  (%.1f bps)\n",
                (rmse_bps < 50.0) ? "PASS" : "FAIL", rmse_bps);
    std::printf("  Max err < 100 bps: %s  (%.1f bps)\n",
                (max_bps < 100.0) ? "PASS" : "FAIL", max_bps);
    std::printf("  Feller condition:  %s\n",
                result.params.feller_satisfied() ? "PASS" : "WARN");
    std::printf("  Converged:         %s\n",
                result.converged ? "YES" : "NO");

    check_feller(result.params);
}

} // namespace heston
