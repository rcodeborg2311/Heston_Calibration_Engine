#!/usr/bin/env python3
"""
validate_vs_quantlib.py — Compare Heston engine prices vs QuantLib reference.

QuantLib's Heston engine uses the analytic integration formula and is a
well-established reference implementation.  This script:

  1. Defines a grid of (K, T) test cases with known Heston parameters.
  2. Prices via QuantLib (if installed).
  3. Prices via our Python Heston FFT implementation.
  4. Reports discrepancies.

Usage:
    pip install QuantLib-Python
    python3 python/validate_vs_quantlib.py

If QuantLib is not installed, prints the FFT prices only.
"""

import math
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Reuse Heston FFT from fetch_data.py
# ---------------------------------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from fetch_data import heston_call_price, bs_implied_vol, bs_call

# Test parameters
PARAMS = dict(v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=-0.7)
S, r, q = 100.0, 0.05, 0.02

TEST_CASES = [
    # (K, T)
    (90.0,  0.25),
    (95.0,  0.25),
    (100.0, 0.25),
    (105.0, 0.25),
    (110.0, 0.25),
    (90.0,  0.50),
    (100.0, 0.50),
    (110.0, 0.50),
    (100.0, 1.00),
    (90.0,  1.00),
    (110.0, 1.00),
]

# ---------------------------------------------------------------------------
# QuantLib pricing (optional)
# ---------------------------------------------------------------------------

def quantlib_heston_price(K: float, T: float) -> Optional[float]:
    """Price via QuantLib's Heston analytic engine."""
    try:
        import QuantLib as ql

        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        exercise = ql.EuropeanExercise(today + int(T * 365))
        payoff   = ql.PlainVanillaPayoff(ql.Option.Call, K)
        option   = ql.VanillaOption(payoff, exercise)

        spot_h   = ql.QuoteHandle(ql.SimpleQuote(S))
        rts_h    = ql.YieldTermStructureHandle(
                       ql.FlatForward(today, r, ql.Actual365Fixed()))
        div_h    = ql.YieldTermStructureHandle(
                       ql.FlatForward(today, q, ql.Actual365Fixed()))

        v0, kappa, theta, xi, rho = (PARAMS['v0'], PARAMS['kappa'],
                                      PARAMS['theta'], PARAMS['xi'], PARAMS['rho'])
        process = ql.HestonProcess(rts_h, div_h, spot_h,
                                   v0, kappa, theta, xi, rho)
        model   = ql.HestonModel(process)
        engine  = ql.AnalyticHestonEngine(model)
        option.setPricingEngine(engine)

        return option.NPV()
    except ImportError:
        return None
    except Exception as e:
        print(f"  QuantLib error at K={K}, T={T}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Heston Engine Validation vs QuantLib")
    print("=" * 70)
    print(f"Parameters: v0={PARAMS['v0']}, kappa={PARAMS['kappa']}, "
          f"theta={PARAMS['theta']}, xi={PARAMS['xi']}, rho={PARAMS['rho']}")
    print(f"Market: S={S}, r={r}, q={q}")
    print()

    ql_available = quantlib_heston_price(100.0, 0.5) is not None

    if ql_available:
        hdr = f"{'K':>8} {'T':>6} {'FFT Price':>12} {'QL Price':>12} "
        hdr += f"{'Diff ($)':>10} {'Diff (bps IV)':>14}"
    else:
        hdr = f"{'K':>8} {'T':>6} {'FFT Price':>12} {'FFT IV (%)':>12}"
        print("  (QuantLib not installed — showing FFT prices only)")
        print("  pip install QuantLib-Python  to enable comparison\n")

    print(hdr)
    print("-" * len(hdr))

    max_diff_bps = 0.0

    for K, T in TEST_CASES:
        fft_price = heston_call_price(S, K, T, r, q, **PARAMS)
        fft_iv    = bs_implied_vol(fft_price, S, K, T, r, q)
        fft_iv_pct = (fft_iv * 100) if fft_iv else float('nan')

        if ql_available:
            ql_price = quantlib_heston_price(K, T)
            if ql_price is None:
                ql_price = float('nan')
                diff_dollar = float('nan')
                diff_bps    = float('nan')
            else:
                diff_dollar = fft_price - ql_price
                # Convert price diff to IV bps via vega
                from fetch_data import bs_vega as _bsvega
                vega = _bsvega(S, K, T, r, q, fft_iv or 0.20)
                diff_bps = (abs(diff_dollar) / vega * 1e4) if vega > 0 else float('nan')
                max_diff_bps = max(max_diff_bps, diff_bps)

            print(f"{K:>8.1f} {T:>6.2f} {fft_price:>12.6f} {ql_price:>12.6f} "
                  f"{diff_dollar:>10.6f} {diff_bps:>13.1f} bps")
        else:
            print(f"{K:>8.1f} {T:>6.2f} {fft_price:>12.6f} {fft_iv_pct:>11.2f}%")

    print("-" * len(hdr))

    if ql_available:
        print(f"\nMax discrepancy vs QuantLib: {max_diff_bps:.2f} bps")
        if max_diff_bps < 5.0:
            print("  PASS: < 5 bps  (excellent agreement)")
        elif max_diff_bps < 20.0:
            print("  WARN: 5-20 bps (acceptable; likely FFT discretisation)")
        else:
            print("  FAIL: > 20 bps (investigation needed)")

    print("\nDone.")


if __name__ == '__main__':
    main()
