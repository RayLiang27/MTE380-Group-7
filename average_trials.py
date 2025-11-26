import os
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog
import matplotlib.pyplot as ploot


def load_csv_auto(path: str) -> pd.DataFrame:
    """
    Load a CSV and ensure:
    - First column is numeric time
    - Columns are named: time, signal_1, signal_2, ...

    Handles both header and no-header files.
    """
    # Try with header row first
    df = pd.read_csv(path)
    # If first column isn't numeric, re-read without header
    if not np.issubdtype(df.iloc[:, 0].dtype, np.number):
        df = pd.read_csv(path, header=None)

    # Rename columns
    n_cols = df.shape[1]
    col_names = ["time"] + [f"signal_{i}" for i in range(1, n_cols)]
    df.columns = col_names[:n_cols]

    return df


def normalize_and_resample(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    - Normalizes time for each DataFrame (start at 0).
    - Creates a common time grid based on all trials.
    - Interpolates each trial onto that grid.
    - Returns a DataFrame with averaged signals.

    Assumes:
    - Each df has columns: time, signal_1, signal_2, ...
    - All dfs have the same number of signal columns.
    """

    # Normalize time (start at 0) for each trial
    norm_dfs = []
    for df in dfs:
        df = df.copy()
        df["time"] = df["time"] - df["time"].iloc[0]
        norm_dfs.append(df)

    # Determine common time grid:
    # - Use the smallest max time across trials
    # - Use the smallest median dt as the time step
    max_times = [df["time"].max() for df in norm_dfs]
    t_end = min(max_times)

    dts = [np.median(np.diff(df["time"].values)) for df in norm_dfs]
    dt = min(dts)

    # Build common time vector
    common_time = np.arange(0.0, t_end + 0.5 * dt, dt)

    # Interpolate all signals from each trial onto common_time
    signal_cols = [c for c in norm_dfs[0].columns if c != "time"]
    n_signals = len(signal_cols)
    n_trials = len(norm_dfs)
    n_points = len(common_time)

    # 3D array: (trials, time, signals)
    all_data = np.zeros((n_trials, n_points, n_signals))

    for i, df in enumerate(norm_dfs):
        t = df["time"].values
        for j, col in enumerate(signal_cols):
            y = df[col].values
            # 1D interpolation for each signal
            all_data[i, :, j] = np.interp(common_time, t, y)

    # Average over trials
    avg_data = all_data.mean(axis=0)  # shape: (time, signals)

    # Build output DataFrame
    out_df = pd.DataFrame({"time": common_time})
    for j, col in enumerate(signal_cols):
        out_df[col + "_mean"] = avg_data[:, j]

    return out_df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to lower_snake_case:
    'Time Stamp' -> 'time_stamp'
    'Ball Pixel in Y' -> 'ball_y_px'
    'Servo1 Degrees' -> 'servo1_deg'
    """
    df = df.copy()
    df.columns = [
        c.strip().lower().replace(" ", "_")
        for c in df.columns
    ]
    return df


def compute_ball_lag_samples(servo: np.ndarray,
                             ball: np.ndarray,
                             dt: float) -> tuple[float, float]:
    """
    Compute lag between servo command and ball response using cross-correlation.

    Convention:
    - Positive lag_seconds  -> ball responds AFTER servo (what we expect)
    - Negative lag_seconds  -> ball appears to move BEFORE servo
                               (usually noise / filtering artifact)

    Returns:
        (lag_samples, lag_seconds)
    """
    x = np.asarray(servo, dtype=float)
    y = np.asarray(ball, dtype=float)

    # Remove NaNs if any
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # Detrend by removing mean
    x = x - x.mean()
    y = y - y.mean()

    if len(x) < 2 or len(y) < 2:
        raise ValueError("Not enough data points for correlation.")

    # Full cross-correlation
    c = np.correlate(x, y, mode="full")
    lags = np.arange(-len(y) + 1, len(x))

    best_lag = lags[np.argmax(c)]

    # Our convention: positive result = ball lags servo
    ball_lags_servo_samples = -best_lag
    ball_lags_servo_seconds = ball_lags_servo_samples * dt

    return ball_lags_servo_samples, ball_lags_servo_seconds


def compute_lag_crosscorr(x, y, dt, max_lag_s=0.3):
    """
    x = servo signal (cause)
    y = ball signal (effect)
    """
    x = x - np.mean(x)
    y = y - np.mean(y)

    # Normalize magnitude so correlation is shape-based
    if np.std(x) > 0:
        x = x / np.std(x)
    if np.std(y) > 0:
        y = y / np.std(y)

    # Full correlation
    c = np.correlate(x, y, mode="full")
    lags = np.arange(-len(y) + 1, len(x))

    # Restrict lag window
    max_lag_samples = int(max_lag_s / dt)
    mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)

    c = c[mask]
    lags = lags[mask]

    best_lag = lags[np.argmax(c)]

    # Interpret following the convention:
    # positive → ball lags servo
    ball_lag_samples = -best_lag
    ball_lag_seconds = ball_lag_samples * dt

    return ball_lag_samples, ball_lag_seconds


# def analyze_file(path):
#     print(f"\n=== Analyzing: {os.path.basename(path)} ===")

#     df = pd.read_csv(path)
#     df = normalize_columns(df)

#     required = ["t_s", "ball_y_px", "servo1_deg"]
#     for r in required:
#         if r not in df.columns:
#             raise KeyError(f"Missing column '{r}'")

#     # Normalize time
#     df["time_stamp"] = pd.to_numeric(df["t_s"], errors="coerce")
#     df = df.dropna(subset=["time_stamp"]).sort_values("time_stamp")
#     t = df["time_stamp"].values
#     t_norm = t - t[0]
#     df["time_norm_s"] = t_norm

#     # dt
#     dt = np.median(np.diff(t_norm))
#     print(f"Estimated dt = {dt:.6f} s")

#     # Extract the known movement window around the step at 1.0 s
#     t0, t1 = 0.95, 1.40
#     mask = (df["time_norm_s"] >= t0) & (df["time_norm_s"] <= t1)

#     servo = df["servo1_deg"].values[mask]
#     ball  = df["ball_y_px"].values[mask]

#     if len(servo) < 10 or len(ball) < 10:
#         print("Movement window too small — cannot compute lag.")
#         return

#     # Compute lag
#     lag_samples, lag_seconds = compute_lag_crosscorr(
#         x=servo,
#         y=ball,
#         dt=dt,
#         max_lag_s=0.3
#     )

#     print(f"Lag: {lag_samples:.2f} samples, {lag_seconds*1000:.2f} ms")

#     if lag_seconds > 0:
#         print("Interpretation: ball lags servo (expected motor response).")
#     elif lag_seconds < 0:
#         print("Interpretation: ball leads servo (PID reacting before servo moves).")
#     else:
#         print("Interpretation: zero lag.")

def analyze_file(path):
    print(f"\n=== Analyzing: {os.path.basename(path)} ===")

    df = pd.read_csv(path)
    df = normalize_columns(df)

    required = ["t_s", "ball_y_px", "servo1_deg"]
    for r in required:
        if r not in df.columns:
            raise KeyError(f"Missing column '{r}'")

    # Normalize time
    df["time_stamp"] = pd.to_numeric(df["t_s"], errors="coerce")
    df = df.dropna(subset=["time_stamp"]).sort_values("time_stamp")
    t = df["time_stamp"].values
    t_norm = t - t[0]
    df["time_norm_s"] = t_norm

    # dt
    dt = np.median(np.diff(t_norm))
    print(f"Estimated dt = {dt:.6f} s")

    # Extract the known movement window around the step at 1.0 s
    t0, t1 = 0.9, 1.50
    mask = (df["time_norm_s"] >= t0) & (df["time_norm_s"] <= t1)

    t_plot = df["time_norm_s"].values[mask]
    servo = df["servo1_deg"].values[mask]
    ball  = df["ball_y_px"].values[mask]

    plot_df = pd.DataFrame({
    "time": t_plot,
    "servo_deg": servo,
    "ball_y_px": ball
    })

    # Plot both on the same figure
    plot_df.plot(x="time", y=["ball_y_px"], title=os.path.basename(path))
    ploot.show()

    if len(servo) < 10 or len(ball) < 10:
        print("Movement window too small — cannot compute lag.")
        return

    # Compute lag
    lag_samples, lag_seconds = compute_lag_crosscorr(
        x=servo,
        y=ball,
        dt=dt,
        max_lag_s=0.3
    )

    print(f"Lag: {lag_samples:.2f} samples, {lag_seconds*1000:.2f} ms")

    if lag_seconds > 0:
        print("Interpretation: ball lags servo (expected motor response).")
    elif lag_seconds < 0:
        print("Interpretation: ball leads servo (PID reacting before servo moves).")
    else:
        print("Interpretation: zero lag.")


def main():
    # Hide the root Tk window
    root = Tk()
    root.withdraw()

    # Ask user to select CSV files (e.g., 3 trials)
    paths = filedialog.askopenfilenames(
        title="Select CSV files for trials",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if not paths:
        print("No files selected. Exiting.")
        return

    for path in paths:
        try:
            analyze_file(path)
        except Exception as e:
            print(f"Error analyzing {os.path.basename(path)}: {e}")

    # # Load all trials
    # dfs = [load_csv_auto(p) for p in paths]

    # # Sanity check: all trials same number of columns
    # n_signals_set = {df.shape[1] for df in dfs}
    # if len(n_signals_set) != 1:
    #     print("Error: Not all CSVs have the same number of columns.")
    #     print("Column counts:", n_signals_set)
    #     return

    # # Normalize, resample, and average
    # avg_df = normalize_and_resample(dfs)

    # # Save result next to the first file
    # first_path = paths[0]
    # base_dir = os.path.dirname(first_path)
    # base_name = os.path.splitext(os.path.basename(first_path))[0]
    # out_name = base_name + "_averaged.csv"
    # out_path = os.path.join(base_dir, out_name)

    # avg_df.to_csv(out_path, index=False)
    # print(f"\nAveraged data saved to:\n  {out_path}")


if __name__ == "__main__":
    main()
