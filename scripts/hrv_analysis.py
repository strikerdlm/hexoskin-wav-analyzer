"""
Heart-Rate Variability (HRV) Analysis for the Valquiria dataset.

This script automatically:
1. Loads physiological data (prefers the merged SQLite database, with CSV fallback).
2. Derives RR-intervals from heart-rate (bpm) samples.
3. Computes HRV metrics in three domains:
   • Time-domain  – mean RR, SDNN, RMSSD, NN50, pNN50, etc.
   • Frequency-domain – VLF, LF, HF power, LF/HF ratio (Welch PSD).
   • Non-linear – Poincaré SD1, SD2, SD2/SD1.
4. Generates visualisations:
   • RR-interval time-series
   • Welch power spectral density with VLF/LF/HF bands shaded
   • Poincaré plot
5. Saves a single CSV summary plus PNG images into  `hrv_results/`.

Usage
-----
    python hrv_analysis.py

Dependencies
------------
    pip install hrvanalysis scipy numpy pandas matplotlib seaborn
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.signal import welch

import sys, types, importlib

# -----------------------------------------------------------------------------
# Compatibility patch: ensure `astropy.stats.LombScargle` exists (removed in astropy>=6)
# -----------------------------------------------------------------------------
try:
    from astropy.stats import LombScargle as _LS_test  # noqa: F401
except ImportError:
    try:
        from astropy.timeseries import LombScargle as _LS  # type: ignore
        import astropy
        stats_mod = types.ModuleType("astropy.stats")
        stats_mod.LombScargle = _LS
        sys.modules["astropy.stats"] = stats_mod
        astropy.stats = stats_mod  # type: ignore
        print("ℹ️ Patched astropy.stats.LombScargle alias for hrvanalysis compatibility.")
    except ImportError:
        # Will raise later when hrvanalysis tries to import
        pass

# -----------------------------------------------------------------------------
# External package for HRV feature extraction
from hrvanalysis import (
    get_time_domain_features,
    get_frequency_domain_features,
    get_poincare_plot_features,
)

# Local utility for loading the dataset
try:
    from load_data import load_database_data, load_csv_data  # type: ignore
except ImportError:
    # Fallback absolute import when running from repository root
    from Data.joined_data.scripts.load_data import (  # type: ignore
        load_database_data,
        load_csv_data,
    )


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _setup_plotting() -> None:
    """Apply a consistent Seaborn/Matplotlib style."""
    sns.set(style="whitegrid")
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def _rr_from_hr(hr_series: pd.Series) -> np.ndarray:
    """Convert heart-rate (bpm) to RR-intervals (ms), drop implausible beats."""
    rr_ms = 60_000.0 / hr_series.astype(float)
    rr_ms = rr_ms.replace([np.inf, -np.inf], np.nan).dropna()
    # Filter physiologically implausible intervals (<300 ms or >2000 ms)
    rr_ms = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
    return rr_ms.values


def _plot_rr_timeseries(rr_ms: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure(figsize=(14, 4))
    plt.plot(np.arange(len(rr_ms)), rr_ms / 1000.0, linewidth=0.7)
    plt.title(f"RR-interval Time-Series – {title}")
    plt.xlabel("Beat index")
    plt.ylabel("RR (s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_psd(rr_ms: np.ndarray, title: str, out_path: Path) -> None:
    """Compute & plot Welch PSD with HRV frequency bands shaded."""
    # Build tachogram cumulative time axis
    t = np.cumsum(rr_ms) / 1000.0  # seconds
    fs_interp = 4.0  # Hz – recommended for HRV
    resampled_time = np.arange(0.0, t[-1], 1 / fs_interp)
    # Interpolate using cubic for smoothness
    f_interp = interp1d(t, rr_ms, kind="cubic", fill_value="extrapolate")
    resampled_rr = f_interp(resampled_time)
    # Detrend
    resampled_rr -= np.mean(resampled_rr)
    # Welch PSD
    freqs, psd = welch(resampled_rr, fs=fs_interp, nperseg=256)

    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, psd, lw=1.2)
    # Shade standard HRV bands
    plt.axvspan(0.0033, 0.04, color="yellow", alpha=0.3, label="VLF")
    plt.axvspan(0.04, 0.15, color="green", alpha=0.3, label="LF")
    plt.axvspan(0.15, 0.40, color="red", alpha=0.3, label="HF")
    plt.title(f"Welch PSD – {title}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (ms²/Hz) – log scale")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_poincare(rr_ms: np.ndarray, title: str, sd1: float, sd2: float, out_path: Path) -> None:
    rr1, rr2 = rr_ms[:-1], rr_ms[1:]
    plt.figure(figsize=(6, 6))
    plt.scatter(rr1, rr2, s=8, alpha=0.4)
    lims = [min(rr1.min(), rr2.min()), max(rr1.max(), rr2.max())]
    plt.plot(lims, lims, "k--", alpha=0.6)
    plt.xlabel("RRₙ (ms)")
    plt.ylabel("RRₙ₊₁ (ms)")
    plt.title(f"Poincaré Plot – {title}\nSD1={sd1:.1f} ms, SD2={sd2:.1f} ms")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _analyse_segment(
    df: pd.DataFrame,
    subject: str,
    sol: str,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    rr_ms = _rr_from_hr(df["heart_rate [bpm]"])
    if len(rr_ms) < 50:
        print(f"⚠️ Insufficient RR-intervals for {subject} Sol {sol} (n={len(rr_ms)}) – skipping")
        return None

    # ---------------- HRV feature extraction ----------------
    time_feats = get_time_domain_features(rr_ms)
    freq_feats = get_frequency_domain_features(rr_ms)
    nl_feats = get_poincare_plot_features(rr_ms)

    # ---------------- Visualisations ----------------
    ts_path = output_dir / f"rr_ts_{subject}_Sol{sol}.png"
    psd_path = output_dir / f"welch_psd_{subject}_Sol{sol}.png"
    pc_path = output_dir / f"poincare_{subject}_Sol{sol}.png"

    _plot_rr_timeseries(rr_ms, f"{subject} – Sol {sol}", ts_path)
    _plot_psd(rr_ms, f"{subject} – Sol {sol}", psd_path)
    _plot_poincare(rr_ms, f"{subject} – Sol {sol}", nl_feats["sd1"], nl_feats["sd2"], pc_path)

    # Merge all metrics into a single dict
    result = {
        "subject": subject,
        "Sol": sol,
        **time_feats,
        **freq_feats,
        **nl_feats,
    }
    return result


# -----------------------------------------------------------------------------
# Main entry-point
# -----------------------------------------------------------------------------

def main() -> None:
    _setup_plotting()

    # 1. Load data (SQLite first, CSV fallback)
    df = load_database_data("merged_data.db")
    if df is None:
        csvs = load_csv_data()
        if not csvs:
            print("❌ No data available for HRV analysis – aborting.")
            return
        df = pd.concat(csvs.values(), ignore_index=True)
        print(f"Loaded {len(df):,} rows from CSV fallback.")
    else:
        print(f"Loaded {len(df):,} rows from merged database.")

    # Ensure required column exists
    if "heart_rate [bpm]" not in df.columns:
        print("❌ Column 'heart_rate [bpm]' not found – cannot compute HRV.")
        return

    output_dir = Path(__file__).parent / "hrv_results"
    output_dir.mkdir(exist_ok=True)

    metrics: list[Dict[str, Any]] = []

    # 2. Iterate by subject & Sol where available
    if "subject" in df.columns:
        grouping = ["subject"]
    else:
        grouping = []
    if "Sol" in df.columns:
        grouping.append("Sol")

    if grouping:
        group_iter = df.groupby(grouping)
        print(f"Processing HRV for {len(group_iter)} groups ({' × '.join(grouping)}).")
        for keys, segment in group_iter:
            subject, sol = (keys if len(grouping) == 2 else (keys, "All"))
            print(f"→ {subject} Sol {sol}: {len(segment):,} rows")
            res = _analyse_segment(segment, str(subject), str(sol), output_dir)
            if res:
                metrics.append(res)
    else:
        print("Processing entire dataset as a single segment.")
        res_all = _analyse_segment(df, "All", "All", output_dir)
        if res_all:
            metrics.append(res_all)

    # 3. Save combined metrics
    if metrics:
        metrics_df = pd.DataFrame(metrics)
        csv_path = output_dir / "hrv_metrics_summary.csv"
        metrics_df.to_csv(csv_path, index=False)
        print(f"✅ HRV metrics saved to {csv_path}")
    else:
        print("⚠️ No HRV metrics were computed – check data availability and filters.")


if __name__ == "__main__":
    main() 