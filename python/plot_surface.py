#!/usr/bin/env python3
"""
plot_surface.py — 3D implied-volatility surface visualisation.

Reads results/calibration_output.csv (produced by heston_demo).
Produces:
  results/vol_surface.png        — side-by-side 3D market vs model surface
  results/vol_smile_per_expiry.png — 2×2 smile plot per expiry

Usage:
    python3 python/plot_surface.py [results/calibration_output.csv]
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import griddata


RESULTS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'results')
DEFAULT_CSV  = os.path.join(RESULTS_DIR, 'calibration_output.csv')
SURFACE_PNG  = os.path.join(RESULTS_DIR, 'vol_surface.png')
SMILE_PNG    = os.path.join(RESULTS_DIR, 'vol_smile_per_expiry.png')


# ---------------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Columns: strike, expiry, market_iv, model_iv
    # Some rows may have empty market_iv (grid-only rows)
    df['market_iv'] = pd.to_numeric(df['market_iv'], errors='coerce')
    df['model_iv']  = pd.to_numeric(df['model_iv'],  errors='coerce')
    df = df.dropna(subset=['model_iv'])
    return df


# ---------------------------------------------------------------------------
# 3D surface plot
# ---------------------------------------------------------------------------

def plot_3d_surface(df: pd.DataFrame, spot: float = 100.0):
    """Side-by-side 3D plots: market IV (scatter) and model IV (mesh)."""

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle('Heston Model Calibration — SPY / Synthetic Options',
                 fontsize=14, fontweight='bold', y=1.01)

    # Market and model rows
    market_df = df[df['market_iv'].notna()].copy()
    model_df  = df.copy()

    # Build a grid for the model surface
    K_min, K_max = model_df['strike'].min(), model_df['strike'].max()
    T_min, T_max = model_df['expiry'].min(), model_df['expiry'].max()

    K_grid = np.linspace(K_min, K_max, 50)
    T_grid = np.linspace(T_min, T_max, 30)
    KK, TT = np.meshgrid(K_grid, T_grid)

    # Interpolate model IV onto grid
    points = model_df[['strike', 'expiry']].values
    vals   = model_df['model_iv'].values
    ZZ_model = griddata(points, vals, (KK, TT), method='linear')

    # Moneyness X-axis
    XX = KK / spot

    vmin = np.nanmin([market_df['market_iv'].min() if len(market_df) > 0 else 0.10,
                      np.nanmin(ZZ_model)]) * 0.95
    vmax = np.nanmax([market_df['market_iv'].max() if len(market_df) > 0 else 0.40,
                      np.nanmax(ZZ_model)]) * 1.05

    # ---- Left: market IV ----
    ax1 = fig.add_subplot(121, projection='3d')
    if len(market_df) > 0:
        sc = ax1.scatter(market_df['strike'] / spot,
                         market_df['expiry'],
                         market_df['market_iv'],
                         c=market_df['market_iv'],
                         cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                         s=40, zorder=5)
        plt.colorbar(sc, ax=ax1, shrink=0.5, label='Implied Vol')
    ax1.set_xlabel('Moneyness K/S', labelpad=8)
    ax1.set_ylabel('Expiry (years)', labelpad=8)
    ax1.set_zlabel('Implied Vol', labelpad=8)
    ax1.set_title('Market Implied Vol', pad=10)
    ax1.view_init(elev=25, azim=-60)

    # ---- Right: model IV ----
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(XX, TT, ZZ_model,
                             cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                             alpha=0.85, linewidth=0)
    plt.colorbar(surf, ax=ax2, shrink=0.5, label='Implied Vol')
    ax2.set_xlabel('Moneyness K/S', labelpad=8)
    ax2.set_ylabel('Expiry (years)', labelpad=8)
    ax2.set_zlabel('Implied Vol', labelpad=8)
    ax2.set_title('Heston Model Implied Vol', pad=10)
    ax2.view_init(elev=25, azim=-60)

    # Footer with RMSE
    if len(market_df) > 0:
        merged = market_df.merge(
            model_df[['strike', 'expiry', 'model_iv']],
            on=['strike', 'expiry'], how='left', suffixes=('', '_m'))
        if 'model_iv_m' in merged.columns:
            err = (merged['model_iv'] - merged['model_iv_m']).dropna()
        else:
            err = (merged['market_iv'] - merged['model_iv']).dropna()
        rmse_bps = math.sqrt((err ** 2).mean()) * 1e4 if len(err) > 0 else float('nan')
        fig.text(0.5, -0.02,
                 f'RMSE = {rmse_bps:.1f} bps | {len(market_df)} market options',
                 ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(SURFACE_PNG, dpi=150, bbox_inches='tight')
    print(f"Saved: {SURFACE_PNG}")
    plt.close()


# ---------------------------------------------------------------------------
# Per-expiry smile plot
# ---------------------------------------------------------------------------

def plot_smiles_per_expiry(df: pd.DataFrame, spot: float = 100.0):
    """2×2 grid of smile plots, one per expiry."""
    market_df = df[df['market_iv'].notna()].copy()
    model_df  = df.copy()

    # Get up to 4 distinct expiries from market data (or model grid)
    if len(market_df) > 0:
        expiries = sorted(market_df['expiry'].unique())[:4]
    else:
        expiries = sorted(model_df['expiry'].unique())
        # Pick 4 representative ones
        idx = np.round(np.linspace(0, len(expiries) - 1, 4)).astype(int)
        expiries = [expiries[i] for i in idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Heston Model — Vol Smile per Expiry\n'
                 '(dots = market, line = Heston model)',
                 fontsize=13, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (ax, T) in enumerate(zip(axes.flat, expiries)):
        color = colors[i % len(colors)]

        # Model: continuous line over strike range
        m_rows = model_df[np.abs(model_df['expiry'] - T) < 0.01].sort_values('strike')
        if len(m_rows) > 0:
            ax.plot(m_rows['strike'] / spot, m_rows['model_iv'] * 100,
                    color=color, linewidth=2.0, label='Heston model')

        # Market: dots
        mk_rows = market_df[np.abs(market_df['expiry'] - T) < 0.015].sort_values('strike')
        if len(mk_rows) > 0:
            ax.scatter(mk_rows['strike'] / spot, mk_rows['market_iv'] * 100,
                       color=color, s=50, zorder=5, label='Market')

            # Error bars (±1 bps visual guide)
            if 'model_iv' in mk_rows.columns:
                rmse = math.sqrt(((mk_rows['market_iv'] - mk_rows['model_iv']) ** 2).mean())
            else:
                rmse = float('nan')
            ax.set_title(f'T = {T:.3f} yr  |  RMSE = {rmse*1e4:.1f} bps',
                         fontsize=10)
        else:
            ax.set_title(f'T = {T:.3f} yr', fontsize=10)

        ax.set_xlabel('Moneyness K/S', fontsize=9)
        ax.set_ylabel('Implied Vol (%)', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.75, 1.25)

    plt.tight_layout()
    plt.savefig(SMILE_PNG, dpi=150, bbox_inches='tight')
    print(f"Saved: {SMILE_PNG}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        print("Run ./build/src/heston_demo first to generate calibration output.")
        sys.exit(1)

    print(f"Loading {csv_path}...")
    df = load_csv(csv_path)
    print(f"  {len(df)} rows loaded.")

    # Try to infer spot from data (ATM moneyness = 1)
    spot = 100.0  # default

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Plotting 3D surface...")
    plot_3d_surface(df, spot)

    print("Plotting per-expiry smiles...")
    plot_smiles_per_expiry(df, spot)

    print("\nDone! Check results/")


if __name__ == '__main__':
    main()
