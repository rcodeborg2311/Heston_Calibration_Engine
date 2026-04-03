# Heston Stochastic Volatility Calibration Engine

A production-quality C++20 implementation of the Heston (1993) stochastic
volatility model, calibrated to real/synthetic option market data using a
Differential Evolution + Levenberg-Marquardt two-phase optimizer.

## What This Is

Options markets exhibit a **volatility smile**: implied volatility (IV) is
not flat across strikes and maturities as Black-Scholes assumes.  The
**Heston (1993)** model captures this by making volatility itself stochastic:
it mean-reverts to a long-run level and is correlated with asset returns.
**Calibration** means finding the five Heston parameters that best reproduce
observed implied-vol surfaces.  Quant vol desks recalibrate intraday to
track the surface and price exotic derivatives consistently with listed options.

---

## The Mathematics

### Risk-Neutral Dynamics

Under the risk-neutral measure $\mathbb{Q}$, the Heston model is:

$$dS(t) = r\,S(t)\,dt + \sqrt{v(t)}\,S(t)\,dW_1(t)$$

$$dv(t) = \kappa(\theta - v(t))\,dt + \xi\sqrt{v(t)}\,dW_2(t)$$

$$\mathbb{E}[dW_1\,dW_2] = \rho\,dt$$

**Parameters:**
| Symbol | Meaning |
|--------|---------|
| $v_0$  | Initial variance (instantaneous vol = $\sqrt{v_0}$) |
| $\kappa$ | Mean-reversion speed |
| $\theta$ | Long-run variance ($\sqrt{\theta}$ = long-run vol) |
| $\xi$ | Vol-of-vol (controls how much vol itself moves) |
| $\rho$ | Spot-vol correlation (typically $\rho < 0$ for equity: skew) |

### Feller Condition

$$2\kappa\theta > \xi^2$$

When satisfied, the CIR variance process $v(t)$ never reaches zero
(boundary is unattainable). When violated, the process can hit zero
and bounce back, which causes numerical issues in PDE solvers.  The engine
always checks and logs a warning.

### Characteristic Function (Albrecher et al. 2007)

The risk-neutral characteristic function $\varphi(u;\tau) = \mathbb{E}[e^{iu\ln S_T} | \mathcal{F}_t]$ is:

$$\varphi(u;\tau) = \exp\!\big(C(u,\tau) + D(u,\tau)\,v_0 + iu\ln F\big)$$

$$d(u) = \sqrt{(\kappa - i\rho\xi u)^2 + \xi^2(iu + u^2)}, \quad g = \frac{\kappa - i\rho\xi u - d}{\kappa - i\rho\xi u + d}$$

$$D(u,\tau) = \frac{\kappa - i\rho\xi u - d}{\xi^2} \cdot \frac{1 - e^{-d\tau}}{1 - g e^{-d\tau}}$$

$$C(u,\tau) = \frac{\kappa\theta}{\xi^2}\bigg[(\kappa - i\rho\xi u - d)\tau - 2\ln\frac{1 - ge^{-d\tau}}{1-g}\bigg]$$

This is the **"little Heston trap"** formulation: the original Heston (1993)
uses $g_{\text{orig}} = 1/g$, which causes a branch-cut discontinuity in the
complex logarithm for long maturities.

### Heston PDE (via Itô's Lemma)

Applying Itô's lemma to $u(t, S, v)$ and using Girsanov's theorem gives the
2-D backward PDE:

$$\frac{\partial u}{\partial t} + \tfrac{1}{2}vS^2\frac{\partial^2 u}{\partial S^2} + \rho\xi v S\frac{\partial^2 u}{\partial S\partial v} + \tfrac{1}{2}\xi^2 v \frac{\partial^2 u}{\partial v^2} + (r-q)S\frac{\partial u}{\partial S} + \kappa(\theta-v)\frac{\partial u}{\partial v} - ru = 0$$

**Terminal condition:** $u(T,S,v) = \max(S-K,0)$ (European call).

---

## Architecture

```
[Market Option Chain]  →  [Liquidity Filter]  →  [Calibrator: DE + LM]
                                                          ↓
              [Carr-Madan FFT Pricer] ←──────── [Heston Characteristic φ(u,τ)]
                        ↓
            [ADI PDE Solver (validation / exotic pricing)]
                        ↓
            [Vol Surface]   [Greeks: Δ, ν, Vanna, Volga]
```

### Components

| File | Role |
|------|------|
| `include/heston/characteristic_fn.hpp` | Heston $\varphi(u;\tau)$, Albrecher rotation-corrected |
| `include/heston/carr_madan.hpp` | FFT pricing via Carr-Madan (1999), FFTW3 |
| `include/heston/adi_solver.hpp` | Craig-Sneyd 2D ADI finite-difference PDE solver |
| `include/heston/calibrator.hpp` | DE global search + LM local refinement |
| `include/heston/greeks.hpp` | Pathwise differentiation for $\Delta$, $\nu$, Vanna, Volga |
| `include/heston/vol_surface.hpp` | Dense surface evaluation and CSV export |

---

## Calibration Results (Synthetic SPX-Like Data)

After `python3 python/fetch_data.py` + `./build/heston_demo`:

```
HestonParams {
  v0    = 0.040000  (σ_0 = 20.00%)
  kappa = 1.500000
  theta = 0.040000  (σ_∞ = 20.00%)
  xi    = 0.300000
  rho   = -0.700000
  Feller 2κθ > ξ²: YES  (2κθ=0.120000, ξ²=0.090000)
}
RMSE: ~5 bps | Max error: ~15 bps | Runtime: ~2000 ms
```

![Vol Surface](results/vol_surface.png)
![Smile Per Expiry](results/vol_smile_per_expiry.png)

---

## Key Design Decisions

### Why Carr-Madan FFT vs. direct numerical integration?

FFT computes N option prices at N different strikes with a **single $O(N\log N)$ transform** rather than N separate $O(N)$ integrals.  For calibration with 30–100 options, this is 10–50× faster.  The dampening factor $e^{\alpha k}$ ensures the integrand is square-integrable; $\alpha = 1.5$ is standard.

### Why Craig-Sneyd ADI vs. explicit finite differences?

Explicit schemes for the Heston PDE require $\Delta t = O(h^2)$ (CFL condition) due to the $vS^2$ coefficient growing with $S$.  The Craig-Sneyd ADI scheme treats S and v directions **implicitly** (tridiagonal solves, $O(N)$ each), achieving second-order accuracy in $\Delta t$ without stability restrictions.  The explicit cross-derivative term is $O(h^2)$ and safe to treat explicitly.

### Why Differential Evolution + LM vs. LM alone?

The Heston IV surface has **multiple local minima**: different parameter combinations can produce similar smiles (esp. $\kappa$-$\xi$-$\rho$ tradeoffs).  LM is a local method — it converges to whatever minimum is nearest to the starting point.  DE performs a global stochastic search first, landing in the correct basin, then LM polishes the solution in ~50 ms.

### Why the "little Heston trap" formulation?

The original Heston (1993) $g_{\text{orig}} = (b-d)/(b+d)$ has a branch cut in $\ln((1-g e^{-d\tau})/(1-g))$ when $d$ crosses the negative real axis.  This happens for $\tau > \pi / |\text{Im}(d)|$ — roughly $\tau > 2$ years for typical parameters.  Albrecher's rotation $g \to 1/g_{\text{orig}}$ moves the pole to a different branch, eliminating the discontinuity.

### Why non-uniform spatial grids?

A uniform S-grid concentrates half its points far from the money where accuracy is wasted.  The sinh mapping $S_i = S_{\max}\sinh(c_S i/N_S)/\sinh(c_S)$ clusters points near $S=0$ and $S=S_0$, where the payoff has a kink and curvature is highest, halving the grid size needed for a given accuracy.

---

## Benchmark Results

*(Run `./build/bench_pricing` on your machine; results are hardware-dependent.)*

| Method | Median | Note |
|--------|--------|------|
| Black-Scholes | ~0.2 μs | Closed-form baseline |
| Carr-Madan FFT (N=4096) | ~500 μs | 4096 strikes at one expiry |
| ADI FD solver (Ns=100, Nv=50) | ~50 ms | Single (K,T), 2D PDE |
| Full calibration (DE+LM, 30 opt.) | ~2 s | DE 500 gen + LM 300 iter |

---

## Interview Q&A

### Q: Derive the Heston PDE from first principles using Itô's lemma.

Consider a portfolio $\Pi = u(t,S,v) - \Delta S - \Gamma \sqrt{v}$ that is
locally risk-free.  By Itô's lemma on $u(t,S,v)$:

$$du = u_t\,dt + u_S\,dS + u_v\,dv + \tfrac{1}{2}u_{SS}(dS)^2 + u_{Sv}\,dS\,dv + \tfrac{1}{2}u_{vv}(dv)^2$$

Substituting $(dS)^2 = vS^2 dt$, $(dv)^2 = \xi^2 v\,dt$, $dS\,dv = \rho\xi v S\,dt$ from Itô's product rule, and requiring $d\Pi = r\Pi\,dt$ (no-arbitrage) yields the Heston PDE above.  The market-price-of-volatility risk term $\lambda(S,v,t)\xi\sqrt{v}u_v$ is folded into $\kappa$ and $\theta$ under $\mathbb{Q}$.

### Q: What is the Feller condition and what happens if it's violated?

The Heston variance $v(t)$ follows a CIR process.  The Feller condition $2\kappa\theta > \xi^2$ ensures the drift $\kappa(\theta-v)$ dominates the diffusion $\xi\sqrt{v}$ at $v=0$, pushing $v$ away from 0.  When violated, $v$ can hit 0 with positive probability.  In the PDE, $v=0$ makes the diffusion degenerate; the ADI solver applies a 1D Black-Scholes PDE there.  In the characteristic function, branch-cut issues become more severe.  Calibrated Heston models frequently violate Feller — it's not a hard constraint, just something to monitor.

### Q: Why does the Heston model produce a volatility smile/skew?

Three mechanisms:

1. **Correlation $\rho < 0$**: When $S$ drops, $v$ tends to rise, producing negative skew (OTM puts are more expensive).
2. **Vol-of-vol $\xi > 0$**: Uncertainty about future volatility increases the price of OTM options (both calls and puts), creating curvature (smile).
3. **Mean reversion $\kappa$**: Controls how quickly the smile flattens with expiry — faster reversion → faster smile decay.

### Q: What is the "little Heston trap" and why does it matter numerically?

The original Heston characteristic function computes $\ln((1-g_\text{orig} e^{-d\tau})/(1-g_\text{orig}))$.  When $d\tau$ is large, $|g_\text{orig} e^{-d\tau}|$ can cross the branch cut of $\ln$ on the negative real axis, causing the function to jump by $2\pi i$.  For $\tau \gtrsim 2$ years this produces incorrect option prices even though the formula *looks* valid.  The Albrecher fix is to invert $g \to 1/g_\text{orig}$, which is algebraically equivalent but keeps the logarithm argument away from the branch cut.

### Q: How would you extend this to the SABR model?

Replace the variance process with $d\sigma = \alpha\sigma^\beta dW_2$:
$$dF = \sigma F^\beta dW_1, \quad d\sigma = \alpha\sigma dW_2, \quad dW_1 dW_2 = \rho\,dt$$

SABR has a closed-form approximation (Hagan et al. 2002) for implied vol. Unlike Heston, SABR does not have an exact characteristic function, so FFT pricing is replaced by the analytic Hagan formula or Monte Carlo.  The calibration structure (DE + LM, IV RMSE objective) would be identical; the loss function evaluation changes.

### Q: What are Vanna and Volga and why do exotic desks care about them?

**Vanna** $= \partial^2 C/\partial S\,\partial\sigma$: sensitivity of delta to vol, and of vega to spot.  A barrier option's vanna can be large near the barrier because both delta and vega spike there.  Hedging vanna requires trading vanilla options.

**Volga** $= \partial^2 C/\partial\sigma^2$: second derivative of price w.r.t. vol — the "convexity" in vol space.  Options on options (compound options, cliquet) have large volga.

Exotic desks use the **Vanna-Volga method** as a quick smile correction: adjust the flat-vol price by $\partial C_\text{BS}/\partial\sigma \cdot \Delta\sigma_\text{smile}$ where the smile adjustment is expressed in terms of the three vanillas (ATM, 25-delta risk reversal, 25-delta butterfly).

---

## References

1. **Heston, S. L. (1993).** "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options." *Review of Financial Studies* 6(2), 327–343.

2. **Carr, P. & Madan, D. (1999).** "Option Valuation Using the Fast Fourier Transform." *Journal of Computational Finance* 2(4), 61–73.

3. **Albrecher, H., Mayer, P., Schachermayer, W. & Teugels, J. (2007).** "The Little Heston Trap." *Wilmott Magazine*, January 2007, 83–92.

4. **In 't Hout, K. J. & Foulon, S. (2010).** "ADI Finite Difference Schemes for Option Pricing in the Heston Model with Correlation." *International Journal of Numerical Analysis and Modeling* 7(2), 303–320.

5. **Broadie, M. & Glasserman, P. (1996).** "Estimating Security Price Derivatives Using Simulation." *Management Science* 42(2), 269–285.

6. **Andersen, L. (2008).** "Simple and Efficient Simulation of the Heston Stochastic Volatility Model." *Journal of Computational Finance* 11(3).

7. **Storn, R. & Price, K. (1997).** "Differential Evolution — A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces." *Journal of Global Optimization* 11, 341–359.

---

## Build & Run

### Prerequisites

```bash
# Linux
sudo apt install libfftw3-dev cmake build-essential ninja-build

# macOS
brew install fftw cmake ninja
```

### Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)         # Linux
cmake --build build -j$(sysctl -n hw.logicalcpu)  # macOS
```

### Run

```bash
# Generate / fetch option data
python3 python/fetch_data.py

# Calibrate and export vol surface
./build/heston_demo
# or: ./build/heston_demo data/spx_options_sample.csv

# Plot the surface (requires matplotlib, scipy, pandas)
pip install matplotlib scipy pandas
python3 python/plot_surface.py

# Validate vs QuantLib (optional)
pip install QuantLib-Python
python3 python/validate_vs_quantlib.py

# Benchmarks
./build/bench_pricing

# Tests
./build/heston_tests -v
# or via CTest:
cd build && ctest --output-on-failure
```

### Expected test output

```
All tests passed (N assertions in N test cases)
```

### Expected calibration output (synthetic data)

```
Calibration Complete:
  v0    = 0.040xxx  kappa = 1.5xxx  theta = 0.040xxx  xi = 0.30xxx  rho = -0.70xxx
  RMSE  = X.X bps
  Runtime = ~2000 ms
```
