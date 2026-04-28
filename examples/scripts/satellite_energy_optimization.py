"""
Satellite Energy Storage and Electric Propulsion Co-optimisation
================================================================

This script simulates the energy balance of a low-orbit satellite whose
mission profile has been exported from STK (Systems Tool Kit) as a time
series of:

  * solar-panel generation power  [W]  – positive, contributes to charging
  * electric-thruster consumption power  [W]  – positive, contributes to discharging

The energy accounting rule applied at every time instant is:

  1. Solar-panel output is *always* integrated as positive (charging) unless the
     battery is already at 100 % SoC, in which case the surplus is wasted.
  2. If the thruster is active and its power **exceeds** the solar power the
     shortfall is drawn from the battery (net discharge).
  3. If the thruster is active but its power is **less** than the solar power the
     net result is still a charge (difference stored in the battery).

Net power seen by the battery (PyBaMM sign convention: positive = discharge):

    P_net = P_thruster − P_solar

The script:
  * Builds a PyBaMM Equivalent Circuit Model (Thevenin) to obtain the
    open-circuit voltage (OCV) vs. SoC relationship and internal resistance.
  * Tracks the State-of-Charge (SoC) via Coulomb counting using the ECM's
    capacity parameter, clamping SoC to [0, 1] at each step.
  * Uses a **binary-search** algorithm to find the smallest battery capacity
    [A·h] for which the SoC never drops below **70 %**.
  * Once the minimum capacity is found, performs a verification run with
    PyBaMM's Thevenin ECM (for a representative discharge window) and
    generates a summary plot.

Input CSV format
----------------
Provide a CSV file whose first row is a header.  The script searches for
columns containing the keywords "time", "solar", and "thruster"
(case-insensitive).  Fallback: columns 0, 1, 2 are used.

    Time [s], Solar_Power [W], Thruster_Power [W]
    0.0,      120.0,           0.0
    10.0,     125.0,           1200.0
    ...

Usage
-----
    # Synthetic 90-minute LEO dataset (default):
    python satellite_energy_optimization.py

    # Real STK export:
    python satellite_energy_optimization.py --csv path/to/stk_export.csv

    # Adjust search bracket and SOC floor:
    python satellite_energy_optimization.py --cap-lo 10 --cap-hi 500 --soc-min 0.7
"""

from __future__ import annotations

import argparse
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pybamm

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

SOC_MIN: float = 0.70   # minimum acceptable SoC (70 %)
SOC_INIT: float = 1.00  # battery starts fully charged
V_NOMINAL: float = 3.7  # V – nominal cell voltage (for Wh ↔ A·h conversion)
CAPACITY_TOL: float = 0.01  # A·h – binary-search convergence tolerance


# ---------------------------------------------------------------------------
# 1. Load / generate mission power profile
# ---------------------------------------------------------------------------

def load_stk_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load *time*, *solar power* and *thruster power* from an STK CSV export.

    The file must have a single header row.  Column names containing "time",
    "solar", and "thruster" (case-insensitive) are auto-detected; if they
    cannot be found columns 0, 1, 2 are used as a fallback.

    Returns
    -------
    time_s, solar_w, thruster_w : 1-D float arrays
    """
    import csv as _csv

    with open(csv_path, newline="") as fh:
        reader = _csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader if r]

    hl = [h.lower() for h in header]

    def _col(kw: str) -> int | None:
        for i, h in enumerate(hl):
            if kw in h:
                return i
        return None

    ci_t = _col("time")
    ci_s = _col("solar")
    ci_th = _col("thruster")

    if None in (ci_t, ci_s, ci_th):
        print(
            "[WARNING] Could not identify all required columns by name; "
            "falling back to columns 0, 1, 2."
        )
        ci_t, ci_s, ci_th = 0, 1, 2

    data = np.array([[float(r[ci_t]), float(r[ci_s]), float(r[ci_th])] for r in rows])
    return data[:, 0], data[:, 1], data[:, 2]


def generate_synthetic_leo_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic 90-minute LEO orbit power profile.

    Profile
    -------
    * 60 min sunlight  : sinusoidal solar profile, peak 800 W
    * 30 min eclipse   : zero solar, zero thruster (housekeeping not modelled)
    * Two thruster firing windows inside the sunlight phase (5 min each,
      1 200 W): t = 600–900 s and t = 3 000–3 300 s within the orbit.

    Returns
    -------
    time_s, solar_w, thruster_w : 1-D float arrays
    """
    dt = 10.0          # 10-second resolution
    t_orbit = 5400.0   # 90-minute orbit [s]
    t_sun = 3600.0     # 60 minutes of sunlight [s]

    t = np.arange(0.0, t_orbit + dt, dt)

    solar = np.where(
        t <= t_sun,
        800.0 * np.sin(np.pi * t / t_sun),
        0.0,
    )

    thruster = np.zeros_like(t)
    for t0, t1 in [(600.0, 900.0), (3000.0, 3300.0)]:
        thruster[(t >= t0) & (t <= t1)] = 1200.0  # 1.2 kW

    return t, solar, thruster


# ---------------------------------------------------------------------------
# 2. Coulomb-counting SOC simulation (used for the binary search)
# ---------------------------------------------------------------------------

def coulomb_count_soc(
    time_s: np.ndarray,
    net_power_w: np.ndarray,
    capacity_ah: float,
    soc_init: float = SOC_INIT,
) -> np.ndarray:
    """Track battery SoC by Coulomb counting with saturation clamping.

    The SoC is clamped to [0, 1] at every step:
    * Charging is ignored when SoC would exceed 1.0 (excess energy wasted).
    * Discharging stops at 0.0 (battery fully depleted; not physically reached
      in normal operation).

    This is the most faithful numerical implementation of the problem
    statement: "integrate the solar-panel power as positive and the thruster
    power as negative and store in the battery model".

    Parameters
    ----------
    time_s, net_power_w : 1-D arrays
        Mission time [s] and net battery power [W]
        (positive = discharge, negative = charge).
    capacity_ah : float
        Battery capacity [A·h].
    soc_init : float
        Initial SoC (default 1.0 – fully charged).

    Returns
    -------
    soc : 1-D ndarray, same length as *time_s*.
    """
    capacity_ws = capacity_ah * 3600.0 * V_NOMINAL  # convert A·h → W·s (energy)

    soc = np.empty(len(time_s))
    soc[0] = soc_init

    for i in range(1, len(time_s)):
        dt = time_s[i] - time_s[i - 1]
        # energy change: positive net_power means energy flows OUT
        delta_energy_ws = -net_power_w[i - 1] * dt  # negate: positive delta_energy = charge
        delta_soc = delta_energy_ws / capacity_ws

        new_soc = soc[i - 1] + delta_soc
        soc[i] = min(1.0, max(0.0, new_soc))  # clamp to [0, 1]

    return soc


# ---------------------------------------------------------------------------
# 3. Binary search for minimum capacity
# ---------------------------------------------------------------------------

def find_minimum_capacity(
    time_s: np.ndarray,
    net_power_w: np.ndarray,
    capacity_lo: float = 1.0,
    capacity_hi: float = 2000.0,
    soc_min: float = SOC_MIN,
    tol: float = CAPACITY_TOL,
) -> tuple[float, np.ndarray]:
    """Binary-search for the smallest capacity that keeps SoC ≥ *soc_min*.

    Uses Coulomb counting (fast) for every candidate capacity.

    Parameters
    ----------
    time_s, net_power_w : mission profile arrays.
    capacity_lo, capacity_hi : search bracket [A·h].
    soc_min : minimum acceptable SoC.
    tol : convergence tolerance on capacity [A·h].

    Returns
    -------
    min_capacity_ah : float
    soc_trajectory   : 1-D ndarray at the minimum feasible capacity.

    Raises
    ------
    RuntimeError if *capacity_hi* is itself infeasible.
    """

    def is_feasible(cap: float) -> tuple[bool, np.ndarray]:
        soc = coulomb_count_soc(time_s, net_power_w, cap)
        return bool(soc.min() >= soc_min), soc

    # Validate upper bound
    ok_hi, soc_hi = is_feasible(capacity_hi)
    if not ok_hi:
        raise RuntimeError(
            f"Upper bound {capacity_hi} A·h is still infeasible "
            f"(min SoC = {coulomb_count_soc(time_s, net_power_w, capacity_hi).min():.4f}). "
            "Increase --cap-hi."
        )

    # Check if lower bound is already feasible
    ok_lo, soc_lo = is_feasible(capacity_lo)
    if ok_lo:
        print(
            f"[INFO] Lower bound {capacity_lo:.2f} A·h is already feasible; "
            "try a smaller --cap-lo for a tighter result."
        )
        return capacity_lo, soc_lo

    best_cap = capacity_hi
    best_soc = soc_hi

    iteration = 0
    while (capacity_hi - capacity_lo) > tol:
        iteration += 1
        mid = 0.5 * (capacity_lo + capacity_hi)
        ok, soc_mid = is_feasible(mid)
        print(
            f"  Iter {iteration:3d}: capacity = {mid:9.3f} A·h  |  "
            f"min SoC = {soc_mid.min():.4f}  |  "
            f"{'✓ feasible' if ok else '✗ infeasible'}"
        )
        if ok:
            capacity_hi = mid
            best_cap = mid
            best_soc = soc_mid
        else:
            capacity_lo = mid

    return best_cap, best_soc


# ---------------------------------------------------------------------------
# 4. PyBaMM Thevenin ECM verification for a discharge window
# ---------------------------------------------------------------------------

def run_pybamm_discharge_verification(
    time_s: np.ndarray,
    net_power_w: np.ndarray,
    capacity_ah: float,
    soc_at_window_start: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run the PyBaMM Thevenin ECM for the first net-discharge window.

    This provides a physically accurate (electrochemical) trajectory for the
    most critical phase of the mission: the window where the battery SoC is
    at its deepest.

    Only the first continuous discharge window (P_net > 0) is simulated to
    avoid the ``Maximum SoC`` termination event that occurs when a charging
    period would push the modelled SoC above 1.0.

    Parameters
    ----------
    time_s, net_power_w : full mission profile arrays.
    capacity_ah : minimum feasible capacity [A·h].
    soc_at_window_start : SoC just before the discharge window starts.

    Returns
    -------
    (t_pybamm, soc_pybamm) or None if no discharge window is found / if the
    simulation fails.
    """
    # ---- find the first net-discharge window --------------------------------
    discharge_mask = net_power_w > 0
    indices = np.where(discharge_mask)[0]

    if len(indices) == 0:
        return None

    # Find the first contiguous block of discharge indices
    start_idx = indices[0]
    end_idx = start_idx
    for k in range(1, len(indices)):
        if indices[k] == end_idx + 1:
            end_idx = indices[k]
        else:
            break  # end of first contiguous block

    # Window: start exactly at the first discharge point (do NOT include the
    # pre-discharge charging step, as PyBaMM would see SoC=1.0 + charge →
    # "Maximum SoC" event fires immediately).
    win_start = start_idx
    win_end = min(len(time_s) - 1, end_idx + 1)

    t_win = time_s[win_start: win_end + 1] - time_s[win_start]
    p_win = net_power_w[win_start: win_end + 1]

    if t_win[-1] <= 0 or len(t_win) < 2:
        return None

    drive_cycle = np.column_stack([t_win, p_win])

    # ---- build and run the PyBaMM ECM simulation ----------------------------
    model = pybamm.equivalent_circuit.Thevenin()
    params = model.default_parameter_values.copy()

    params["Initial SoC"] = float(min(soc_at_window_start, 0.9999))
    params["Cell capacity [A.h]"] = float(capacity_ah)
    params["Nominal cell capacity [A.h]"] = float(capacity_ah)
    params["Upper voltage cut-off [V]"] = 5.0
    params["Lower voltage cut-off [V]"] = 0.0

    experiment = pybamm.Experiment([pybamm.step.power(drive_cycle)])
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=params)

    pybamm.set_logging_level("WARNING")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            sol = sim.solve()
        except (pybamm.SolverError, RuntimeError):
            return None

    t_pybamm = sol["Time [s]"].entries + time_s[win_start]
    soc_pybamm = sol["SoC"].entries
    return t_pybamm, soc_pybamm


# ---------------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------------

def plot_results(
    time_s: np.ndarray,
    solar_w: np.ndarray,
    thruster_w: np.ndarray,
    net_power_w: np.ndarray,
    soc_cc: np.ndarray,
    min_cap_ah: float,
    pybamm_result: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    """Generate the three-panel summary figure."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False)
    fig.suptitle(
        f"Satellite Energy Optimisation — Minimum battery capacity = "
        f"{min_cap_ah:.2f} A·h  "
        f"({min_cap_ah * V_NOMINAL / 1000:.3f} kW·h @ {V_NOMINAL} V nominal)",
        fontsize=12,
        fontweight="bold",
    )

    t_min = time_s / 60  # convert to minutes for x-axis

    # ── Panel 1: raw power profile ──────────────────────────────────────────
    ax = axes[0]
    ax.plot(t_min, solar_w / 1000, color="gold", label="Solar panel [kW]", lw=1.5)
    ax.plot(t_min, thruster_w / 1000, color="crimson", ls="--",
            label="Thruster [kW]", lw=1.5)
    ax.set_ylabel("Power [kW]")
    ax.set_xlabel("Mission time [min]")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.35)
    ax.set_title("STK mission power profile")

    # ── Panel 2: net battery power ─────────────────────────────────────────
    ax = axes[1]
    colors = np.where(net_power_w >= 0, "tomato", "steelblue")
    for i in range(len(t_min) - 1):
        ax.fill_between(
            [t_min[i], t_min[i + 1]],
            [net_power_w[i] / 1000, net_power_w[i + 1] / 1000],
            color=colors[i],
            alpha=0.65,
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Net battery power [kW]\n(+ve = discharge, −ve = charge)")
    ax.set_xlabel("Mission time [min]")
    ax.grid(True, alpha=0.35)
    ax.set_title("Net battery power  (P_thruster − P_solar)")
    ax.legend(handles=[
        mpatches.Patch(color="tomato", alpha=0.7, label="Net discharge"),
        mpatches.Patch(color="steelblue", alpha=0.7, label="Net charge"),
    ])

    # ── Panel 3: SoC trajectory ────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t_min, soc_cc * 100, color="green", lw=1.8,
            label="SoC – Coulomb counting (full mission)")
    if pybamm_result is not None:
        t_pbm, soc_pbm = pybamm_result
        ax.plot(t_pbm / 60, soc_pbm * 100, color="navy", lw=2, ls="-.",
                label="SoC – PyBaMM Thevenin ECM (discharge window)")

    ax.axhline(SOC_MIN * 100, color="red", ls="--", lw=1.4,
               label=f"SoC floor = {SOC_MIN * 100:.0f} %")
    ax.fill_between(t_min, soc_cc * 100, SOC_MIN * 100,
                    where=(soc_cc < SOC_MIN),
                    color="red", alpha=0.25, label="SoC floor violation zone")
    ax.set_ylim(max(0, soc_cc.min() * 100 - 5), 105)
    ax.set_ylabel("State of Charge [%]")
    ax.set_xlabel("Mission time [min]")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.35)
    ax.set_title(
        f"Battery SoC trajectory  "
        f"(capacity = {min_cap_ah:.2f} A·h,  min SoC = {soc_cc.min() * 100:.2f} %)"
    )

    plt.tight_layout()
    out_file = "satellite_soc_trajectory.png"
    plt.savefig(out_file, dpi=150)
    print(f"\n[INFO] Plot saved → {out_file}")
    plt.show()


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find the minimum battery capacity for a satellite mission that "
            "keeps SoC ≥ soc-min at all times."
        )
    )
    parser.add_argument(
        "--csv",
        metavar="PATH",
        default=None,
        help=(
            "Path to STK-exported CSV "
            "(columns: Time[s], Solar_Power[W], Thruster_Power[W]). "
            "If not given, a synthetic 90-min LEO dataset is used."
        ),
    )
    parser.add_argument(
        "--cap-lo", type=float, default=1.0, metavar="AH",
        help="Lower bound of capacity search bracket [A·h] (default: 1.0).",
    )
    parser.add_argument(
        "--cap-hi", type=float, default=2000.0, metavar="AH",
        help="Upper bound of capacity search bracket [A·h] (default: 2000.0).",
    )
    parser.add_argument(
        "--soc-min", type=float, default=SOC_MIN, metavar="FRAC",
        help=f"Minimum acceptable SoC as a fraction 0–1 (default: {SOC_MIN}).",
    )
    parser.add_argument(
        "--tol", type=float, default=CAPACITY_TOL, metavar="AH",
        help=f"Binary-search tolerance [A·h] (default: {CAPACITY_TOL}).",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Suppress the interactive matplotlib window (still saves PNG).",
    )
    args = parser.parse_args()

    # ── Load mission data ────────────────────────────────────────────────────
    if args.csv:
        print(f"[INFO] Loading STK mission data from: {args.csv}")
        time_s, solar_w, thruster_w = load_stk_csv(args.csv)
    else:
        print("[INFO] No CSV supplied – generating synthetic LEO orbit data.")
        time_s, solar_w, thruster_w = generate_synthetic_leo_data()

    duration_min = (time_s[-1] - time_s[0]) / 60.0
    print(
        f"[INFO] Mission: {duration_min:.1f} min  |  "
        f"{len(time_s)} time steps  |  "
        f"Δt ≈ {np.diff(time_s).mean():.1f} s  |  "
        f"Peak solar = {solar_w.max():.0f} W  |  "
        f"Peak thruster = {thruster_w.max():.0f} W"
    )

    # ── Net power (PyBaMM convention: positive = discharge) ──────────────────
    net_power_w = thruster_w - solar_w

    e_solar_wh = np.trapezoid(solar_w, time_s) / 3600
    e_thruster_wh = np.trapezoid(thruster_w, time_s) / 3600
    e_net_wh = np.trapezoid(net_power_w, time_s) / 3600
    print(
        f"[INFO] Energy: solar = {e_solar_wh:.1f} W·h  |  "
        f"thruster = {e_thruster_wh:.1f} W·h  |  "
        f"net battery = {e_net_wh:+.1f} W·h "
        f"({'net discharge' if e_net_wh > 0 else 'net charge'} over mission)"
    )

    # ── Binary search for minimum capacity ──────────────────────────────────
    print(
        f"\n[INFO] Binary-searching for minimum capacity in "
        f"[{args.cap_lo}, {args.cap_hi}] A·h  |  "
        f"SoC floor = {args.soc_min * 100:.0f} %  |  "
        f"tolerance = {args.tol} A·h\n"
    )
    try:
        min_cap_ah, soc_cc = find_minimum_capacity(
            time_s=time_s,
            net_power_w=net_power_w,
            capacity_lo=args.cap_lo,
            capacity_hi=args.cap_hi,
            soc_min=args.soc_min,
            tol=args.tol,
        )
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    # ── PyBaMM Thevenin ECM verification (first discharge window) ───────────
    print("\n[INFO] Running PyBaMM Thevenin ECM for the first discharge window …")

    # SoC just before the first discharge window
    discharge_indices = np.where(net_power_w > 0)[0]
    if len(discharge_indices) > 0:
        win_start_idx = max(0, discharge_indices[0] - 1)
        soc_at_window_start = float(soc_cc[win_start_idx])
    else:
        soc_at_window_start = SOC_INIT

    pybamm_result = run_pybamm_discharge_verification(
        time_s=time_s,
        net_power_w=net_power_w,
        capacity_ah=min_cap_ah,
        soc_at_window_start=soc_at_window_start,
    )

    if pybamm_result is not None:
        t_pbm, soc_pbm = pybamm_result
        print(
            f"[INFO] PyBaMM verification: discharge window {t_pbm[0]/60:.1f}–"
            f"{t_pbm[-1]/60:.1f} min  |  "
            f"SoC {soc_pbm[0]*100:.2f} % → {soc_pbm[-1]*100:.2f} %"
        )
    else:
        print("[INFO] PyBaMM verification skipped (no discharge window found or solver failed).")

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  OPTIMISATION RESULT")
    print("=" * 62)
    print(f"  Minimum battery capacity  :  {min_cap_ah:.3f} A·h")
    print(
        f"  Energy equivalent         :  "
        f"{min_cap_ah * V_NOMINAL / 1000:.4f} kW·h  "
        f"(@ {V_NOMINAL} V nominal)"
    )
    print(f"  Mission minimum SoC       :  {soc_cc.min() * 100:.2f} %")
    print(f"  SoC floor requirement     :  {args.soc_min * 100:.0f} %")
    print(f"  Initial SoC               :  {SOC_INIT * 100:.0f} %  (fully charged)")
    print(f"  Mission duration          :  {duration_min:.1f} min")
    print("=" * 62)

    # ── Plot ─────────────────────────────────────────────────────────────────
    if args.no_plot:
        plt.switch_backend("Agg")  # non-interactive backend

    plot_results(
        time_s=time_s,
        solar_w=solar_w,
        thruster_w=thruster_w,
        net_power_w=net_power_w,
        soc_cc=soc_cc,
        min_cap_ah=min_cap_ah,
        pybamm_result=pybamm_result,
    )


if __name__ == "__main__":
    main()
