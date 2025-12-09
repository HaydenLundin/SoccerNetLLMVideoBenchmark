#!/usr/bin/env python3
"""
Model Comparison Analysis Script
Compares Qwen 7B, Phi-3.5 Vision, and SoccerNet Professional Annotations

This script analyzes predictions from multiple vision-language models against
professional SoccerNet annotations. Phi3.5 only covers the first 6 games
(12 video halves: video_0 through video_11).

Features:
- Event count comparison across all three sources
- Label distribution analysis
- Temporal heatmap showing when events occur during matches
- Team-based analysis

Usage:
    python compare_models_analysis.py

Requirements:
    pip install pandas matplotlib seaborn numpy
    (Optional for SoccerNet download: pip install soccernet)
"""

import os
import glob
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths - adjust these based on your setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QWEN_DIR = os.path.join(BASE_DIR, "qwenjson")
PHI35_DIR = os.path.join(BASE_DIR, "phi3.5")

# Phi3.5 only covers first 6 games (12 halves = videos 0-11)
PHI35_VIDEO_RANGE = range(0, 12)

# SoccerNet game paths (first 6 games for fair comparison)
# These correspond to videos 0-11 in the prediction files
SELECTED_GAMES_FIRST_6 = [
    "/content/SoccerNet_Data/england_epl/2015-2016/2016-01-13 - 22-45 Chelsea 2 - 2 West Brom",
    "/content/SoccerNet_Data/england_epl/2015-2016/2016-04-09 - 19-30 Manchester City 2 - 1 West Brom",
    "/content/SoccerNet_Data/england_epl/2015-2016/2016-01-03 - 16-30 Crystal Palace 0 - 3 Chelsea",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-10-04 - 19-15 Manchester City 1 - 2 Tottenham",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-11-07 - 18-00 Chelsea 1 - 0 Aston Villa",
]

# All 20 games for full Qwen comparison
SELECTED_GAMES_ALL = [
    "/content/SoccerNet_Data/england_epl/2015-2016/2016-01-13 - 22-45 Chelsea 2 - 2 West Brom",
    "/content/SoccerNet_Data/england_epl/2015-2016/2016-04-09 - 19-30 Manchester City 2 - 1 West Brom",
    "/content/SoccerNet_Data/england_epl/2015-2016/2016-01-03 - 16-30 Crystal Palace 0 - 3 Chelsea",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-10-04 - 19-15 Manchester City 1 - 2 Tottenham",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-11-07 - 18-00 Chelsea 1 - 0 Aston Villa",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-09-12 - 14-45 Everton 3 - 1 Chelsea",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-12-05 - 18-00 Chelsea 1 - 0 Bournemouth",
    "/content/SoccerNet_Data/england_epl/2015-2016/2016-03-02 - 23-00 Liverpool 3 - 0 Manchester City",
    "/content/SoccerNet_Data/england_epl/2015-2016/2016-05-07 - 17-00 Sunderland 3 - 2 Chelsea",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-08-08 - 19-30 Chelsea 2 - 2 Swansea",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-12-19 - 18-00 Chelsea 3 - 1 Sunderland",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-10-31 - 15-45 Chelsea 1 - 3 Liverpool",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-08-29 - 17-00 Manchester City 2 - 0 Watford",
    "/content/SoccerNet_Data/england_epl/2015-2016/2016-03-19 - 18-00 Chelsea 2 - 2 West Ham",
    "/content/SoccerNet_Data/england_epl/2015-2016/2016-04-09 - 17-00 Swansea 1 - 0 Chelsea",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-11-07 - 18-00 Manchester United 2 - 0 West Brom",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-11-21 - 20-30 Manchester City 1 - 4 Liverpool",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-09-19 - 19-30 Manchester City 1 - 2 West Ham",
    "/content/SoccerNet_Data/england_epl/2015-2016/2015-09-12 - 17-00 Crystal Palace 0 - 1 Manchester City",
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_raw_json(raw):
    """Strip ```json fences and parse JSON; return a list of event dictionaries."""
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
        else:
            return []
    except Exception:
        return []


def convert_seconds_to_half_clock(t):
    """Map absolute seconds to SoccerNet-style half + mm:ss clock."""
    if t < 45 * 60:  # First half (0-45 min)
        half = "1"
        clock = f"{int(t//60):02d}:{int(t%60):02d}"
    else:  # Second half (45+ min)
        half = "2"
        t2 = t - 45 * 60
        clock = f"{int(t2//60):02d}:{int(t2%60):02d}"
    return half, clock


def clock_to_minutes(half, clock):
    """Convert half + clock string to total minutes from match start."""
    try:
        parts = clock.split(':')
        minutes = int(parts[0])
        seconds = int(parts[1]) if len(parts) > 1 else 0
        total_minutes = minutes + seconds / 60
        if half == "2":
            total_minutes += 45  # Add first half duration
        return total_minutes
    except:
        return None


def get_time_bin(minutes, bin_size=5):
    """Group minutes into bins (e.g., 0-5, 5-10, etc.)."""
    if minutes is None:
        return None
    return int(minutes // bin_size) * bin_size


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_qwen_predictions(video_indices=None):
    """Load Qwen prediction files."""
    pred_files = sorted(glob.glob(os.path.join(QWEN_DIR, "final_predictions_qwen7b_video_*.json")))

    rows = []
    for file in pred_files:
        # Extract video index from filename
        match = re.search(r'video_(\d+)', file)
        if not match:
            continue
        video_idx = int(match.group(1))

        # Filter by video indices if specified
        if video_indices is not None and video_idx not in video_indices:
            continue

        try:
            with open(file, "r") as f:
                preds_raw = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
            continue

        for entry in preds_raw:
            t = entry.get("time", 0)
            raw_text = entry.get("raw", "")
            events = clean_raw_json(raw_text)

            if not events:
                continue

            half, clock = convert_seconds_to_half_clock(t)

            for event in events:
                rows.append({
                    "source": "Qwen",
                    "video_index": video_idx,
                    "file_path": file,
                    "half": half,
                    "clock": clock,
                    "time_seconds": t,
                    "label": event.get("label"),
                    "team": event.get("team"),
                    "confidence": event.get("confidence"),
                    "details": event.get("details"),
                })

    return pd.DataFrame(rows)


def load_phi35_predictions(video_indices=None):
    """Load Phi3.5 prediction files."""
    pred_files = sorted(glob.glob(os.path.join(PHI35_DIR, "final_predictions_phi3.5_video_*.json")))

    rows = []
    empty_count = 0

    for file in pred_files:
        # Extract video index from filename
        match = re.search(r'video_(\d+)', file)
        if not match:
            continue
        video_idx = int(match.group(1))

        # Filter by video indices if specified
        if video_indices is not None and video_idx not in video_indices:
            continue

        try:
            with open(file, "r") as f:
                preds_raw = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
            continue

        if not preds_raw:
            empty_count += 1
            continue

        for entry in preds_raw:
            t = entry.get("time", 0)
            raw_text = entry.get("raw", "")
            events = clean_raw_json(raw_text)

            if not events:
                continue

            half, clock = convert_seconds_to_half_clock(t)

            for event in events:
                rows.append({
                    "source": "Phi3.5",
                    "video_index": video_idx,
                    "file_path": file,
                    "half": half,
                    "clock": clock,
                    "time_seconds": t,
                    "label": event.get("label"),
                    "team": event.get("team"),
                    "confidence": event.get("confidence"),
                    "details": event.get("details"),
                })

    if empty_count > 0:
        print(f"Note: {empty_count} Phi3.5 files were empty (model produced no outputs)")

    return pd.DataFrame(rows)


def load_soccernet_annotations(game_paths, soccernet_dir=None):
    """
    Load SoccerNet professional annotations.

    Note: This requires SoccerNet data to be downloaded. If not available,
    returns an empty DataFrame with a warning.
    """
    if soccernet_dir is None:
        # Try common locations
        possible_dirs = [
            os.path.join(BASE_DIR, "SoccerNet_Data"),
            "/content/SoccerNet_Data",
            os.path.expanduser("~/SoccerNet_Data"),
        ]
        for d in possible_dirs:
            if os.path.exists(d):
                soccernet_dir = d
                break

    if soccernet_dir is None or not os.path.exists(soccernet_dir):
        print("Warning: SoccerNet data not found. Please download using SoccerNet.Downloader")
        print("Returning empty DataFrame for SoccerNet annotations.")
        return pd.DataFrame()

    records = []
    video_idx = 0  # Track video index (2 halves per game)

    for game_dir in game_paths:
        # Adjust path if needed
        if game_dir.startswith("/content/"):
            game_dir = game_dir.replace("/content/SoccerNet_Data", soccernet_dir)

        season = os.path.basename(os.path.dirname(game_dir))
        game = os.path.basename(game_dir)

        label_path = os.path.join(game_dir, "Labels-v2.json")
        if not os.path.exists(label_path):
            video_idx += 2  # Skip both halves
            continue

        try:
            with open(label_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {label_path}: {e}")
            video_idx += 2
            continue

        for ann in data.get("annotations", []):
            game_time = ann.get("gameTime", "")
            half, clock = None, None

            if " - " in game_time:
                half, clock = game_time.split(" - ")

            # Determine video index based on half
            ann_video_idx = video_idx if half == "1" else video_idx + 1

            position = ann.get("position", [None, None])

            records.append({
                "source": "SoccerNet",
                "video_index": ann_video_idx,
                "season": season,
                "game": game,
                "game_path": game_dir,
                "half": half,
                "clock": clock,
                "label": ann.get("label"),
                "team": ann.get("team", None),
                "player": ann.get("player", None),
                "x": position[0] if isinstance(position, list) else None,
                "y": position[1] if isinstance(position, list) else None,
                "visibility": ann.get("visibility", None),
            })

        video_idx += 2  # Two halves per game

    return pd.DataFrame(records)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def compare_label_counts(df_qwen, df_phi35, df_soccernet):
    """Compare event label counts across all three sources."""

    qwen_counts = df_qwen['label'].value_counts() if not df_qwen.empty else pd.Series(dtype=int)
    phi35_counts = df_phi35['label'].value_counts() if not df_phi35.empty else pd.Series(dtype=int)
    soccernet_counts = df_soccernet['label'].value_counts() if not df_soccernet.empty else pd.Series(dtype=int)

    comparison = pd.concat([qwen_counts, phi35_counts, soccernet_counts], axis=1)
    comparison.columns = ["Qwen", "Phi3.5", "SoccerNet"]
    comparison = comparison.fillna(0).astype(int)
    comparison = comparison.sort_values(by="SoccerNet", ascending=False)

    return comparison


def analyze_by_team(df):
    """Analyze events by team (home/away)."""
    if df.empty:
        return pd.DataFrame()
    return df.groupby(['source', 'team']).size().unstack(fill_value=0)


def create_event_timing_data(df):
    """Prepare data for temporal heatmap."""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df['time_minutes'] = df.apply(
        lambda row: clock_to_minutes(row['half'], row['clock'])
        if pd.notna(row.get('clock')) else (row.get('time_seconds', 0) / 60 if 'time_seconds' in row else None),
        axis=1
    )
    df['time_bin'] = df['time_minutes'].apply(lambda x: get_time_bin(x, 5))

    return df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_label_comparison(comparison_df, title="Event Label Comparison"):
    """Plot bar chart comparing label counts across sources."""
    fig, ax = plt.subplots(figsize=(14, 8))

    comparison_df.head(20).plot(kind='bar', ax=ax, width=0.8)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Event Label", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(title="Source", fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig


def plot_temporal_heatmap(df_qwen, df_phi35, df_soccernet, title="Event Timing Heatmap"):
    """
    Create heatmap showing event frequency over time for each source.
    X-axis: Time bins (5-minute intervals)
    Y-axis: Event labels
    """
    # Prepare timing data
    df_qwen = create_event_timing_data(df_qwen)
    df_phi35 = create_event_timing_data(df_phi35)
    df_soccernet = create_event_timing_data(df_soccernet)

    # Get all unique labels and time bins
    all_labels = set()
    for df in [df_qwen, df_phi35, df_soccernet]:
        if not df.empty and 'label' in df.columns:
            all_labels.update(df['label'].dropna().unique())
    all_labels = sorted(list(all_labels))

    time_bins = list(range(0, 95, 5))  # 0 to 90 minutes in 5-min bins

    # Create pivot tables for each source
    def create_heatmap_matrix(df, labels, bins):
        if df.empty:
            return pd.DataFrame(0, index=labels, columns=bins)

        pivot = df.groupby(['label', 'time_bin']).size().unstack(fill_value=0)

        # Ensure all labels and bins are present
        for label in labels:
            if label not in pivot.index:
                pivot.loc[label] = 0
        for bin_val in bins:
            if bin_val not in pivot.columns:
                pivot[bin_val] = 0

        return pivot.reindex(index=labels, columns=bins, fill_value=0)

    qwen_matrix = create_heatmap_matrix(df_qwen, all_labels, time_bins)
    phi35_matrix = create_heatmap_matrix(df_phi35, all_labels, time_bins)
    soccernet_matrix = create_heatmap_matrix(df_soccernet, all_labels, time_bins)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    # Common settings
    cmap = 'YlOrRd'

    # Determine common vmax for consistent color scaling
    vmax = max(
        qwen_matrix.values.max() if qwen_matrix.size > 0 else 0,
        phi35_matrix.values.max() if phi35_matrix.size > 0 else 0,
        soccernet_matrix.values.max() if soccernet_matrix.size > 0 else 0,
        1  # Minimum to avoid division issues
    )

    # Plot Qwen heatmap
    sns.heatmap(qwen_matrix, ax=axes[0], cmap=cmap, vmin=0, vmax=vmax,
                cbar_kws={'label': 'Event Count'}, linewidths=0.5)
    axes[0].set_title('Qwen 7B Predictions', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time (minutes)', fontsize=10)
    axes[0].set_ylabel('Event Label', fontsize=10)
    axes[0].set_xticklabels([f'{b}' for b in time_bins], rotation=45)

    # Plot Phi3.5 heatmap
    if not df_phi35.empty:
        sns.heatmap(phi35_matrix, ax=axes[1], cmap=cmap, vmin=0, vmax=vmax,
                    cbar_kws={'label': 'Event Count'}, linewidths=0.5)
    else:
        axes[1].text(0.5, 0.5, 'No Phi3.5 Data Available\n(Empty outputs)',
                     ha='center', va='center', fontsize=12, color='gray',
                     transform=axes[1].transAxes)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
    axes[1].set_title('Phi3.5 Vision Predictions', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time (minutes)', fontsize=10)
    axes[1].set_ylabel('Event Label', fontsize=10)
    if not df_phi35.empty:
        axes[1].set_xticklabels([f'{b}' for b in time_bins], rotation=45)

    # Plot SoccerNet heatmap
    if not df_soccernet.empty:
        sns.heatmap(soccernet_matrix, ax=axes[2], cmap=cmap, vmin=0, vmax=vmax,
                    cbar_kws={'label': 'Event Count'}, linewidths=0.5)
    else:
        axes[2].text(0.5, 0.5, 'SoccerNet Data Not Available\n(Download required)',
                     ha='center', va='center', fontsize=12, color='gray',
                     transform=axes[2].transAxes)
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
    axes[2].set_title('SoccerNet Professional Annotations', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (minutes)', fontsize=10)
    axes[2].set_ylabel('Event Label', fontsize=10)
    if not df_soccernet.empty:
        axes[2].set_xticklabels([f'{b}' for b in time_bins], rotation=45)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_combined_temporal_heatmap(df_qwen, df_phi35, df_soccernet, title="Combined Event Timing Comparison"):
    """
    Create a combined heatmap showing normalized event distribution over time.
    Useful for comparing when different models detect events during the match.
    """
    # Prepare timing data
    df_qwen = create_event_timing_data(df_qwen)
    df_phi35 = create_event_timing_data(df_phi35)
    df_soccernet = create_event_timing_data(df_soccernet)

    time_bins = list(range(0, 95, 5))

    # Count events per time bin for each source
    def count_per_bin(df, bins):
        if df.empty:
            return pd.Series(0, index=bins)
        counts = df.groupby('time_bin').size()
        return counts.reindex(bins, fill_value=0)

    qwen_counts = count_per_bin(df_qwen, time_bins)
    phi35_counts = count_per_bin(df_phi35, time_bins)
    soccernet_counts = count_per_bin(df_soccernet, time_bins)

    # Combine into matrix
    combined = pd.DataFrame({
        'Qwen': qwen_counts,
        'Phi3.5': phi35_counts,
        'SoccerNet': soccernet_counts
    }).T

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 4))

    sns.heatmap(combined, ax=ax, cmap='YlOrRd', annot=True, fmt='d',
                cbar_kws={'label': 'Event Count'}, linewidths=0.5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (minutes from match start)', fontsize=12)
    ax.set_ylabel('Source', fontsize=12)
    ax.set_xticklabels([f'{b}-{b+5}' for b in time_bins], rotation=45)

    # Add vertical line at half-time
    halftime_idx = time_bins.index(45) + 0.5
    ax.axvline(x=halftime_idx, color='white', linewidth=2, linestyle='--')
    ax.text(halftime_idx, -0.3, 'Half-time', ha='center', va='top', fontsize=10, color='black')

    plt.tight_layout()

    return fig


def print_summary_statistics(df_qwen, df_phi35, df_soccernet):
    """Print summary statistics for all sources."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    print(f"\nTotal Events Detected:")
    print(f"  Qwen 7B:     {len(df_qwen):,} events")
    print(f"  Phi3.5:      {len(df_phi35):,} events")
    print(f"  SoccerNet:   {len(df_soccernet):,} events")

    if not df_qwen.empty:
        print(f"\nQwen Video Coverage: videos {df_qwen['video_index'].min()} to {df_qwen['video_index'].max()}")
    if not df_phi35.empty:
        print(f"Phi3.5 Video Coverage: videos {df_phi35['video_index'].min()} to {df_phi35['video_index'].max()}")

    print(f"\nUnique Event Labels:")
    print(f"  Qwen:        {df_qwen['label'].nunique() if not df_qwen.empty else 0}")
    print(f"  Phi3.5:      {df_phi35['label'].nunique() if not df_phi35.empty else 0}")
    print(f"  SoccerNet:   {df_soccernet['label'].nunique() if not df_soccernet.empty else 0}")

    if not df_qwen.empty and 'confidence' in df_qwen.columns:
        conf = df_qwen['confidence'].dropna()
        if len(conf) > 0:
            print(f"\nQwen Confidence Scores:")
            print(f"  Mean: {conf.mean():.2f}, Std: {conf.std():.2f}")
            print(f"  Min: {conf.min():.2f}, Max: {conf.max():.2f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("MODEL COMPARISON ANALYSIS")
    print("Comparing: Qwen 7B vs Phi3.5 Vision vs SoccerNet Professional")
    print("="*70)

    # Load data for first 12 videos (6 games) - Phi3.5 coverage
    print("\n[1/4] Loading prediction data...")

    video_indices_phi35 = list(PHI35_VIDEO_RANGE)

    print(f"  Loading Qwen predictions (videos 0-11)...")
    df_qwen = load_qwen_predictions(video_indices=video_indices_phi35)
    print(f"    Loaded {len(df_qwen):,} Qwen events")

    print(f"  Loading Phi3.5 predictions (videos 0-11)...")
    df_phi35 = load_phi35_predictions(video_indices=video_indices_phi35)
    print(f"    Loaded {len(df_phi35):,} Phi3.5 events")

    print(f"  Loading SoccerNet annotations (first 6 games)...")
    df_soccernet = load_soccernet_annotations(SELECTED_GAMES_FIRST_6)
    print(f"    Loaded {len(df_soccernet):,} SoccerNet events")

    # Print summary
    print_summary_statistics(df_qwen, df_phi35, df_soccernet)

    # Compare label counts
    print("\n[2/4] Comparing event label counts...")
    comparison = compare_label_counts(df_qwen, df_phi35, df_soccernet)
    print("\nEvent Label Comparison (First 6 Games / 12 Videos):")
    print("-" * 50)
    print(comparison.to_string())

    # Generate visualizations
    print("\n[3/4] Generating visualizations...")

    # Plot 1: Label comparison bar chart
    fig1 = plot_label_comparison(comparison, "Event Label Comparison (First 6 Games)")
    fig1.savefig(os.path.join(BASE_DIR, "comparison_labels.png"), dpi=150, bbox_inches='tight')
    print("  Saved: comparison_labels.png")

    # Plot 2: Temporal heatmaps (separate for each source)
    fig2 = plot_temporal_heatmap(df_qwen, df_phi35, df_soccernet,
                                  "Event Timing by Label (First 6 Games)")
    fig2.savefig(os.path.join(BASE_DIR, "comparison_heatmap_by_label.png"), dpi=150, bbox_inches='tight')
    print("  Saved: comparison_heatmap_by_label.png")

    # Plot 3: Combined temporal comparison
    fig3 = plot_combined_temporal_heatmap(df_qwen, df_phi35, df_soccernet,
                                           "Event Frequency Over Time (First 6 Games)")
    fig3.savefig(os.path.join(BASE_DIR, "comparison_heatmap_combined.png"), dpi=150, bbox_inches='tight')
    print("  Saved: comparison_heatmap_combined.png")

    # Save comparison data to CSV
    print("\n[4/4] Saving results...")
    comparison.to_csv(os.path.join(BASE_DIR, "comparison_results.csv"))
    print("  Saved: comparison_results.csv")

    # Save detailed data for further analysis
    if not df_qwen.empty:
        df_qwen.to_csv(os.path.join(BASE_DIR, "qwen_events_first12.csv"), index=False)
        print("  Saved: qwen_events_first12.csv")
    if not df_phi35.empty:
        df_phi35.to_csv(os.path.join(BASE_DIR, "phi35_events_first12.csv"), index=False)
        print("  Saved: phi35_events_first12.csv")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    if df_phi35.empty:
        print("\nNOTE: Phi3.5 produced no event predictions.")
        print("This may be due to model compatibility issues on older GPUs.")
        print("See RETROSPECTIVE_REPORT.md for details on Phi3.5 challenges.")

    return comparison, df_qwen, df_phi35, df_soccernet


if __name__ == "__main__":
    comparison, df_qwen, df_phi35, df_soccernet = main()
