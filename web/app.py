"""
app.py — FastAPI backend for the Heston Calibration Engine web UI.

Wraps the compiled C++ binary (build/heston_demo) and exposes JSON endpoints
consumed by the single-page frontend.

Endpoints:
  GET  /                  → serves index.html
  POST /api/calibrate     → run calibration (synthetic or uploaded CSV)
  GET  /api/surface       → returns last surface CSV as JSON
  GET  /api/health        → liveness check

Usage:
  cd /path/to/calibration_engine
  pip install -r web/requirements.txt
  uvicorn web.app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent          # calibration_engine/
BINARY = ROOT / "build" / "heston_demo"
RESULTS_CSV = ROOT / "results" / "calibration_output.csv"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Heston Calibration Engine", version="1.0.0")

# CORS — allow the Vercel frontend (and localhost for dev) to call this backend.
# Set ALLOWED_ORIGIN env var on Railway to your Vercel URL, e.g.:
#   ALLOWED_ORIGIN=https://heston-engine.vercel.app
_origin = os.getenv("ALLOWED_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_origin] if _origin != "*" else ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "binary_exists": BINARY.exists(),
        "binary_path": str(BINARY),
    }


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)


# ---------------------------------------------------------------------------
# Stdout parser
# ---------------------------------------------------------------------------

def _first_float(pattern: str, text: str, default: float = 0.0) -> float:
    """Return the first float captured by the regex group 1."""
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default


def parse_stdout(stdout: str) -> dict:
    """
    Extract calibration results from heston_demo stdout.

    The binary prints "True parameters:" before calibration (for synthetic mode)
    and then "=== Calibration Complete ===" with the calibrated block.
    We parse from the Calibration Complete section to avoid picking up truth params.
    """
    # Slice stdout to start from the calibration results block
    marker = "=== Calibration Complete ==="
    cal_block = stdout[stdout.find(marker):] if marker in stdout else stdout

    # ---------- Heston params (from calibrated block) ----------
    v0    = _first_float(r'v0\s+=\s+([\d.]+)', cal_block)
    kappa = _first_float(r'kappa\s+=\s+([\d.]+)', cal_block)
    theta = _first_float(r'theta\s+=\s+([\d.]+)', cal_block)
    xi    = _first_float(r'xi\s+=\s+([\d.]+)', cal_block)
    rho   = _first_float(r'rho\s+=\s+(-?[\d.]+)', cal_block)

    # ---------- Quality metrics ----------
    rmse_bps    = _first_float(r'RMSE\s+=\s+([\d.]+)\s+bps', cal_block)
    max_err_bps = _first_float(r'Max error\s+=\s+([\d.]+)\s+bps', cal_block)
    runtime_ms  = _first_float(r'Runtime\s+=\s+([\d.]+)\s+ms', stdout)  # full output for timing
    n_options   = int(_first_float(r'N options\s+=\s+(\d+)', cal_block))
    converged_m = re.search(r'Converged\s+=\s+(\w+)', cal_block)
    converged   = (converged_m.group(1) == "YES") if converged_m else False
    feller_m    = re.search(r'Feller 2.*?:\s+(YES|NO)', cal_block)
    feller      = feller_m.group(1) == "YES" if feller_m else False

    # ---------- Greeks ----------
    greeks_line = re.search(r'Greeks\s*\{(.+?)\}', stdout, re.DOTALL)
    greeks: dict[str, float] = {}
    if greeks_line:
        block = greeks_line.group(1)
        for name in ("delta", "vega", "vanna", "volga", "theta", "rho"):
            m = re.search(rf'{name}=(-?[\d.]+)', block)
            greeks[name] = float(m.group(1)) if m else 0.0

    return {
        "params": {
            "v0": v0, "kappa": kappa, "theta": theta, "xi": xi, "rho": rho,
            "sigma0_pct": round(v0 ** 0.5 * 100, 2),
            "sigma_inf_pct": round(theta ** 0.5 * 100, 2),
            "feller_satisfied": feller,
            "feller_lhs": round(2 * kappa * theta, 6),
            "feller_rhs": round(xi * xi, 6),
        },
        "quality": {
            "rmse_bps": round(rmse_bps, 2),
            "max_error_bps": round(max_err_bps, 2),
            "runtime_ms": round(runtime_ms, 1),
            "converged": converged,
            "n_options": n_options,
            "rmse_pass": rmse_bps < 50.0,
            "max_err_pass": max_err_bps < 100.0,
        },
        "greeks": greeks,
    }


# ---------------------------------------------------------------------------
# Surface CSV reader
# ---------------------------------------------------------------------------

def read_surface_csv(path: Path) -> dict:
    """
    Parse results/calibration_output.csv.

    Columns: strike, expiry, market_iv, model_iv
    Rows with market_iv set   → market observations
    Rows with market_iv empty → model surface grid
    """
    market_pts: list[dict] = []
    surface_pts: list[dict] = []

    if not path.exists():
        return {"market": [], "surface": []}

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            strike  = float(row["strike"])
            expiry  = float(row["expiry"])
            model   = float(row["model_iv"]) if row["model_iv"] else None
            market  = float(row["market_iv"]) if row["market_iv"] else None

            if model is None:
                continue

            if market is not None:
                market_pts.append({
                    "strike": strike, "expiry": expiry,
                    "market_iv": round(market * 100, 4),
                    "model_iv":  round(model  * 100, 4),
                })
            else:
                surface_pts.append({
                    "strike": strike, "expiry": expiry,
                    "model_iv": round(model * 100, 4),
                })

    return {"market": market_pts, "surface": surface_pts}


# ---------------------------------------------------------------------------
# Calibration endpoint
# ---------------------------------------------------------------------------

@app.post("/api/calibrate")
async def calibrate(
    file: Optional[UploadFile] = File(None),
):
    """
    Run Heston calibration.

    If `file` is provided (CSV matching the standard schema), it is used as
    market data.  Otherwise the engine generates synthetic Heston data with
    the default true parameters (S=100, r=5%, q=2%).

    Returns:
      JSON with params, quality metrics, Greeks, and vol surface data.
    """
    if not BINARY.exists():
        raise HTTPException(
            status_code=503,
            detail=f"C++ binary not found at {BINARY}. Run: cmake --build build"
        )

    # Build the command
    cmd = [str(BINARY)]

    # If a CSV was uploaded, write it to a temp file and pass as argument
    tmp_csv_path: Optional[str] = None
    if file is not None and file.filename:
        content = await file.read()
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".csv", delete=False
        ) as tmp:
            tmp.write(content)
            tmp_csv_path = tmp.name
        cmd.append(tmp_csv_path)

    # Run binary from project root so results/ is written correctly
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=120,      # 2-minute timeout
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Calibration timed out (>120s)")
    finally:
        if tmp_csv_path:
            os.unlink(tmp_csv_path)

    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Binary exited with code {proc.returncode}:\n{proc.stderr}"
        )

    stdout = proc.stdout

    try:
        result = parse_stdout(stdout)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse binary output: {exc}\n\nstdout:\n{stdout}"
        )

    surface = read_surface_csv(RESULTS_CSV)
    result["surface"] = surface
    result["raw_stdout"] = stdout

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Surface endpoint (re-reads last run)
# ---------------------------------------------------------------------------

@app.get("/api/surface")
def surface():
    return JSONResponse(content=read_surface_csv(RESULTS_CSV))
