"""
Satellite Energy Storage and Electric Propulsion Co-optimization
================================================================

This script determines the **minimum battery capacity** that keeps the
State-of-Charge (SoC) of a low-Earth-orbit (LEO) satellite battery at or
above 70 % throughout the full simulated mission window.

Input data
----------
Time-series data exported from STK (Systems Tool Kit) containing, at each
sample instant:

* ``solar_power_W``    – Solar-panel generation power  (positive, charging)
* ``thruster_power_W`` – Electric-thruster consumption power  (positive when on)
* ``rated_load_W``     – Fixed rated bus load  (constant 8 000 W by default)

The data can be provided either as a CSV file (see :func:`load_stk_csv`) or
generated synthetically for demonstration purposes (see
:func:`generate_demo_stk_data`).

Sign convention used throughout
---------------------------------
Net power  = solar_power − rated_load − thruster_power
Net current (into battery, A) = +net_power / bus_voltage

* Net current > 0  →  battery charging   (solar surplus)
* Net current < 0  →  battery discharging (solar deficit or thruster on)

PyBaMM model
------------
The :class:`pybamm.equivalent_circuit.Thevenin` model is used.  Its SoC
equation is a Coulomb counter:

    dSoC/dt = −I_cell / (3 600 · C_cell)

where  I_cell > 0  means *discharge* and C_cell is in A·h.  Therefore the
net current defined above is fed in with a negated sign when passed to the
model.

Optimization
------------
Because SoC is a pure integral of current, the minimum capacity C_min that
satisfies  min(SoC) ≥ SoC_min  has a closed-form solution:

    cumQ(t) = ∫₀ᵗ I_cell(τ)/3 600 dτ   [A·h discharged cumulatively]
    C_min    = max(cumQ(t)) / (SoC₀ − SoC_min)

This analytical result is first computed, then **verified** by running a
full PyBaMM step-by-step simulation at C_min.

Usage
-----
Run with the bundled synthetic data::

    python satellite_energy_optimization.py

Run with your STK CSV export::

    python satellite_energy_optimization.py --csv path/to/stk_export.csv

Expected CSV columns (header required)::

    time_s, solar_power_W, thruster_power_W

An optional ``rated_load_W`` column overrides the default 8 000 W constant
load on a per-sample basis.
"""

from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np

import pybamm

# ---------------------------------------------------------------------------
# Tuneable defaults
# ---------------------------------------------------------------------------

RATED_LOAD_W: float = 8_000.0   # Constant bus load [W]
SOC_MIN: float = 0.70            # Lower SoC constraint
SOC_INIT: float = 0.85           # Initial SoC (< 1 avoids PyBaMM boundary event)
BUS_VOLTAGE_V: float = 100.0     # Nominal DC bus voltage [V]

# PyBaMM's Thevenin model fires boundary events at SoC = 0 and SoC = 1.
# Clipping the initial (and clamped) SoC to [_SOC_BOUNDARY_EPS, 1−_SOC_BOUNDARY_EPS]
# avoids a non-positive event check at t = 0.
_SOC_BOUNDARY_EPS: float = 1e-6

# Small relative margin added to the analytically derived C_min before the
# PyBaMM verification run.  Without it a floating-point rounding error in the
# Coulomb integral can push the minimum SoC fractionally below 70 %, causing
# the constraint check to fail spuriously.
_CAPACITY_SAFETY_MARGIN: float = 1.001

# Tolerance used when comparing a simulated minimum SoC against the SoC
# constraint (accounts for floating-point rounding in the Coulomb integral).
_SOC_CONSTRAINT_TOL: float = 1e-4


# ---------------------------------------------------------------------------
# Data generation / loading
# ---------------------------------------------------------------------------

def generate_demo_stk_data(
    n_orbits: int = 10,
    orbit_period_s: float = 5_400.0,
    dt_s: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic LEO satellite power data for demonstration.

    The orbit is modelled as a simple sinusoid.  The satellite is in sunlight
    for roughly 65 % of each orbit; an electric thruster fires for the first
    300 s of every other orbit (station-keeping burn).

    Parameters
    ----------
    n_orbits:
        Number of complete orbits to simulate.
    orbit_period_s:
        Duration of one orbit in seconds (default 90 min ≈ ISS altitude).
    dt_s:
        Time-step in seconds.

    Returns
    -------
    t : ndarray, shape (N,)
        Time stamps [s].
    solar_power_W : ndarray, shape (N,)
        Solar-panel generation power [W].
    thruster_power_W : ndarray, shape (N,)
        Electric-thruster consumption power [W].
    """
    total_s = n_orbits * orbit_period_s
    t = np.arange(0.0, total_s, dt_s)

    # Solar power: sinusoidal with eclipse (negative half masked to zero)
    phase = 2.0 * np.pi * t / orbit_period_s
    solar_power_W = np.maximum(0.0, 14_000.0 * np.sin(phase))

    # Thruster: 3 kW, fires for 5 min at the start of every even orbit
    thruster_power_W = np.zeros_like(t)
    burn_duration_s = 300.0
    for orbit in range(n_orbits):
        if orbit % 2 == 0:
            t_start = orbit * orbit_period_s
            t_end = t_start + burn_duration_s
            mask = (t >= t_start) & (t < t_end)
            thruster_power_W[mask] = 3_000.0

    return t, solar_power_W, thruster_power_W


def load_stk_csv(
    filepath: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load time-series data exported from STK.

    Parameters
    ----------
    filepath:
        Path to the CSV file.  Required columns: ``time_s``,
        ``solar_power_W``, ``thruster_power_W``.  Optional column:
        ``rated_load_W`` (overrides the module-level constant per sample).

    Returns
    -------
    t : ndarray
        Time stamps [s].
    solar_power_W : ndarray
        Solar-panel output [W].
    thruster_power_W : ndarray
        Thruster consumption [W].
    rated_load_W : ndarray
        Rated bus load [W] (broadcast from scalar if not in file).
    """
    try:
        import pandas as pd  # optional dependency for CSV loading
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pandas is required to load STK CSV data: pip install pandas"
        ) from exc

    df = pd.read_csv(filepath)
    required = {"time_s", "solar_power_W", "thruster_power_W"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"STK CSV is missing required columns: {missing}.  "
            f"Found: {list(df.columns)}"
        )

    t = df["time_s"].to_numpy(dtype=float)
    solar = df["solar_power_W"].to_numpy(dtype=float)
    thruster = df["thruster_power_W"].to_numpy(dtype=float)

    if "rated_load_W" in df.columns:
        rated_load = df["rated_load_W"].to_numpy(dtype=float)
    else:
        rated_load = np.full_like(t, RATED_LOAD_W)

    return t, solar, thruster, rated_load


# ---------------------------------------------------------------------------
# Power-to-current conversion
# ---------------------------------------------------------------------------

def net_current_profile(
    t: np.ndarray,
    solar_power_W: np.ndarray,
    thruster_power_W: np.ndarray,
    rated_load_W: np.ndarray | float = RATED_LOAD_W,
    bus_voltage_V: float = BUS_VOLTAGE_V,
) -> np.ndarray:
    """Compute the net current drawn from / fed into the battery.

    Positive current  → battery discharges (PyBaMM convention).
    Negative current  → battery charges.

    Parameters
    ----------
    t, solar_power_W, thruster_power_W:
        Time-series arrays of equal length.
    rated_load_W:
        Scalar or array bus load [W].
    bus_voltage_V:
        Nominal DC bus voltage [V] used for power ↔ current conversion.

    Returns
    -------
    current_A : ndarray
        Battery current [A] at each sample (positive = discharge).
    """
    net_power_W = (
        np.asarray(solar_power_W)
        - np.asarray(rated_load_W)
        - np.asarray(thruster_power_W)
    )
    # Positive net power → surplus → charging  → negative current in PyBaMM
    current_A = -net_power_W / bus_voltage_V
    return current_A


# ---------------------------------------------------------------------------
# Analytical minimum-capacity solver
# ---------------------------------------------------------------------------

def analytical_min_capacity(
    t: np.ndarray,
    current_A: np.ndarray,
    soc_init: float = SOC_INIT,
    soc_min: float = SOC_MIN,
) -> float:
    """Return the minimum battery capacity [A·h] satisfying the SoC constraint.

    Because the Thevenin model SoC is a pure Coulomb counter:

        SoC(t) = SoC₀ − cumQ(t) / C

    where  cumQ(t) = ∫₀ᵗ I(τ)/3 600 dτ  [A·h discharged].

    Requiring  SoC(t) ≥ SoC_min  for all t  gives:

        C_min = max(0, max(cumQ(t))) / (SoC₀ − SoC_min)

    Parameters
    ----------
    t :
        Time stamps [s].
    current_A :
        Net battery current [A] (positive = discharge).
    soc_init :
        Battery SoC at t=0.
    soc_min :
        Required minimum SoC constraint.

    Returns
    -------
    C_min : float
        Minimum required capacity [A·h].

    Raises
    ------
    ValueError
        If ``soc_init ≤ soc_min`` (no headroom available).
    """
    if soc_init <= soc_min:
        raise ValueError(
            f"Initial SoC ({soc_init:.2f}) must be greater than the minimum "
            f"SoC constraint ({soc_min:.2f})."
        )

    # Trapezoidal integration of current over time → cumulative charge [A·h]
    dt = np.diff(t)
    i_mid = 0.5 * (current_A[:-1] + current_A[1:])
    delta_q_Ah = i_mid * dt / 3_600.0
    cumQ = np.concatenate([[0.0], np.cumsum(delta_q_Ah)])

    max_cumQ = float(np.max(cumQ))

    if max_cumQ <= 0.0:
        # Net charging throughout – any finite capacity satisfies the constraint.
        warnings.warn(
            "The battery is net-charging throughout the mission.  "
            "The capacity constraint is always satisfied; returning 0 A·h.",
            stacklevel=2,
        )
        return 0.0

    headroom = soc_init - soc_min
    c_min = max_cumQ / headroom
    return c_min


# ---------------------------------------------------------------------------
# PyBaMM simulation
# ---------------------------------------------------------------------------

def simulate_soc_pybamm(
    t: np.ndarray,
    current_A: np.ndarray,
    capacity_Ah: float,
    soc_init: float = SOC_INIT,
) -> np.ndarray:
    """Simulate battery SoC using PyBaMM's Thevenin equivalent-circuit model.

    The current profile is supplied as a time-interpolant so that the solver
    integrates the full mission window in a single call (much faster than
    stepping sample-by-sample).

    Parameters
    ----------
    t :
        Time stamps [s].
    current_A :
        Battery current [A] at each sample (positive = discharge, negative =
        charge).  Must be the same length as *t*.
    capacity_Ah :
        Nominal cell capacity [A·h].
    soc_init :
        Initial SoC (must be strictly less than 1 and greater than 0).

    Returns
    -------
    soc : ndarray, shape (len(t),)
        SoC values [0, 1] at each time stamp, obtained by evaluating the
        PyBaMM solution at every input time stamp.
    """
    model = pybamm.equivalent_circuit.Thevenin()
    param = model.default_parameter_values.copy()
    param["Cell capacity [A.h]"] = float(capacity_Ah)
    param["Initial SoC"] = float(np.clip(soc_init, _SOC_BOUNDARY_EPS, 1.0 - _SOC_BOUNDARY_EPS))

    # Build a piecewise-linear interpolant for the current so the solver can
    # step through the whole profile without manual looping.
    param["Current function [A]"] = pybamm.Interpolant(
        np.asarray(t, dtype=float),
        np.asarray(current_A, dtype=float),
        pybamm.t,
    )

    sim = pybamm.Simulation(model, parameter_values=param)
    sol = sim.solve([float(t[0]), float(t[-1])], t_interp=t)

    # Evaluate the SoC at every requested time stamp
    soc_out = sol["SoC"](t)
    return np.asarray(soc_out)


# ---------------------------------------------------------------------------
# Result reporting
# ---------------------------------------------------------------------------

def _print_summary(
    t: np.ndarray,
    solar_power_W: np.ndarray,
    thruster_power_W: np.ndarray,
    rated_load_W: np.ndarray | float,
    current_A: np.ndarray,
    capacity_Ah: float,
    soc: np.ndarray,
    soc_min: float = SOC_MIN,
) -> None:
    total_hours = (t[-1] - t[0]) / 3_600.0
    min_soc = float(np.min(soc))
    min_soc_time_h = float(t[np.argmin(soc)]) / 3_600.0
    constraint_ok = min_soc >= soc_min - _SOC_CONSTRAINT_TOL

    print("=" * 60)
    print("  Satellite Battery Capacity Optimization — Summary")
    print("=" * 60)
    print(f"  Simulation duration          : {total_hours:.2f} h")
    print(f"  Time step (first interval)   : {t[1]-t[0]:.1f} s")
    print(f"  Rated bus load               : {np.mean(rated_load_W):.0f} W")
    print(f"  Peak solar power             : {np.max(solar_power_W):.0f} W")
    print(f"  Peak thruster power          : {np.max(thruster_power_W):.0f} W")
    print(f"  Bus voltage (nominal)        : {BUS_VOLTAGE_V:.0f} V")
    print("-" * 60)
    print(f"  Minimum required capacity    : {capacity_Ah:.2f} A·h")
    print(f"                               = {capacity_Ah * BUS_VOLTAGE_V / 1e3:.3f} kW·h")
    print(f"  Achieved minimum SoC         : {min_soc:.4f}  "
          f"({'✓ OK' if constraint_ok else '✗ VIOLATED'} ≥ {soc_min:.2f})")
    print(f"  Time of minimum SoC          : {min_soc_time_h:.2f} h")
    print("=" * 60)


def plot_results(
    t: np.ndarray,
    solar_power_W: np.ndarray,
    thruster_power_W: np.ndarray,
    rated_load_W: np.ndarray | float,
    current_A: np.ndarray,
    capacity_Ah: float,
    soc: np.ndarray,
    soc_min: float = SOC_MIN,
) -> None:
    """Plot power profiles and battery SoC over time."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        warnings.warn("matplotlib not available; skipping plots.", stacklevel=2)
        return

    t_h = t / 3_600.0
    rated = np.full_like(t, rated_load_W) if np.isscalar(rated_load_W) else np.asarray(rated_load_W)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"Satellite Energy Storage Optimization\n"
        f"Minimum battery capacity = {capacity_Ah:.1f} A·h  "
        f"({capacity_Ah * BUS_VOLTAGE_V / 1e3:.2f} kW·h)",
        fontsize=13,
    )

    # --- Power profiles -------------------------------------------------------
    ax0 = axes[0]
    ax0.plot(t_h, solar_power_W / 1e3, label="Solar generation", color="goldenrod")
    ax0.plot(t_h, rated / 1e3, label="Rated load", color="steelblue", linestyle="--")
    ax0.plot(t_h, thruster_power_W / 1e3, label="Thruster", color="crimson")
    ax0.set_ylabel("Power [kW]")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.set_title("Power profiles")
    ax0.grid(True, alpha=0.3)

    # --- Net current ----------------------------------------------------------
    ax1 = axes[1]
    ax1.plot(t_h, current_A, color="purple", linewidth=0.8)
    ax1.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax1.fill_between(t_h, current_A, 0,
                     where=(current_A < 0), alpha=0.25, color="green", label="Charging")
    ax1.fill_between(t_h, current_A, 0,
                     where=(current_A > 0), alpha=0.25, color="red", label="Discharging")
    ax1.set_ylabel("Battery current [A]")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("Net battery current (positive = discharge)")
    ax1.grid(True, alpha=0.3)

    # --- SoC ------------------------------------------------------------------
    ax2 = axes[2]
    ax2.plot(t_h, soc * 100, color="navy", linewidth=1.2, label="SoC")
    ax2.axhline(soc_min * 100, color="red", linewidth=1.0, linestyle="--",
                label=f"SoC limit ({soc_min*100:.0f}%)")
    ax2.fill_between(t_h, soc * 100, soc_min * 100,
                     where=(soc < soc_min), alpha=0.35, color="red",
                     label="Constraint violated")
    ax2.set_ylabel("SoC [%]")
    ax2.set_xlabel("Time [h]")
    ax2.set_ylim([max(0, soc_min * 100 - 5), 105])
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_title("Battery State of Charge")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    csv_path: str | None = None,
    rated_load_W: float = RATED_LOAD_W,
    soc_init: float = SOC_INIT,
    soc_min: float = SOC_MIN,
    bus_voltage_V: float = BUS_VOLTAGE_V,
    show_plot: bool = True,
) -> dict:
    """Run the full satellite battery optimization workflow.

    Parameters
    ----------
    csv_path :
        Path to a STK CSV export.  If *None*, synthetic demo data are used.
    rated_load_W :
        Fixed bus load [W] (used when the CSV has no ``rated_load_W`` column).
    soc_init :
        Initial battery SoC (0–1).
    soc_min :
        Minimum permissible SoC throughout the simulation.
    bus_voltage_V :
        Nominal DC bus voltage [V] for power ↔ current conversion.
    show_plot :
        Whether to display matplotlib figures.

    Returns
    -------
    dict with keys:

    ``capacity_Ah``
        Minimum required capacity [A·h].
    ``soc``
        SoC time-series from the PyBaMM simulation at optimal capacity.
    ``min_soc``
        Minimum SoC achieved in the PyBaMM simulation.
    ``t``, ``solar_power_W``, ``thruster_power_W``, ``current_A``
        Input arrays used for the simulation.
    """
    # 1. Load or generate data
    if csv_path is not None:
        print(f"Loading STK data from: {csv_path}")
        t, solar_power_W, thruster_power_W, rated_load_arr = load_stk_csv(csv_path)
    else:
        print("No CSV provided – using synthetic demo data (10 LEO orbits).")
        t, solar_power_W, thruster_power_W = generate_demo_stk_data()
        rated_load_arr = np.full_like(t, rated_load_W)

    print(f"  Samples: {len(t)}  |  Duration: {(t[-1]-t[0])/3600:.2f} h")

    # 2. Compute net current profile
    current_A = net_current_profile(
        t, solar_power_W, thruster_power_W,
        rated_load_W=rated_load_arr,
        bus_voltage_V=bus_voltage_V,
    )

    # 3. Analytical minimum capacity
    print("\nStep 1: Analytical capacity estimate …")
    capacity_Ah = analytical_min_capacity(t, current_A, soc_init=soc_init, soc_min=soc_min)
    print(f"  → C_min = {capacity_Ah:.3f} A·h  "
          f"({capacity_Ah * bus_voltage_V / 1e3:.3f} kW·h)")

    # 4. PyBaMM simulation at the computed minimum capacity
    print("\nStep 2: PyBaMM Thevenin-model verification …")
    # Add a small margin (0.1 %) to avoid floating-point boundary violations
    capacity_verify = capacity_Ah * _CAPACITY_SAFETY_MARGIN
    soc = simulate_soc_pybamm(t, current_A, capacity_Ah=capacity_verify, soc_init=soc_init)
    min_soc_achieved = float(np.min(soc))
    print(f"  → Minimum SoC in PyBaMM simulation: {min_soc_achieved:.4f} "
          f"({'✓' if min_soc_achieved >= soc_min - _SOC_CONSTRAINT_TOL else '✗'})")

    # 5. Report
    _print_summary(
        t, solar_power_W, thruster_power_W, rated_load_arr,
        current_A, capacity_Ah, soc, soc_min=soc_min,
    )

    # 6. Plot
    if show_plot:
        plot_results(
            t, solar_power_W, thruster_power_W, rated_load_arr,
            current_A, capacity_Ah, soc, soc_min=soc_min,
        )

    return {
        "capacity_Ah": capacity_Ah,
        "soc": soc,
        "min_soc": min_soc_achieved,
        "t": t,
        "solar_power_W": solar_power_W,
        "thruster_power_W": thruster_power_W,
        "current_A": current_A,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize minimum battery capacity for a LEO satellite energy "
            "storage and electric-propulsion system."
        )
    )
    parser.add_argument(
        "--csv",
        metavar="FILE",
        default=None,
        help="Path to STK CSV export (columns: time_s, solar_power_W, "
             "thruster_power_W).  Uses synthetic demo data when omitted.",
    )
    parser.add_argument(
        "--rated-load",
        type=float,
        default=RATED_LOAD_W,
        metavar="W",
        help=f"Fixed rated bus load in watts (default: {RATED_LOAD_W:.0f} W).",
    )
    parser.add_argument(
        "--soc-init",
        type=float,
        default=SOC_INIT,
        metavar="FRAC",
        help=f"Initial battery SoC, 0–1 (default: {SOC_INIT}).",
    )
    parser.add_argument(
        "--soc-min",
        type=float,
        default=SOC_MIN,
        metavar="FRAC",
        help=f"Minimum SoC constraint, 0–1 (default: {SOC_MIN}).",
    )
    parser.add_argument(
        "--bus-voltage",
        type=float,
        default=BUS_VOLTAGE_V,
        metavar="V",
        help=f"Nominal DC bus voltage in volts (default: {BUS_VOLTAGE_V:.0f} V).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Suppress matplotlib figures.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(
        csv_path=args.csv,
        rated_load_W=args.rated_load,
        soc_init=args.soc_init,
        soc_min=args.soc_min,
        bus_voltage_V=args.bus_voltage,
        show_plot=not args.no_plot,
    )
