from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

import pybamm


@dataclass
class OptimizationResult:
    theoretical_capacity_ah: float
    verified_capacity_ah: float
    minimum_soc: float
    capacity_margin_ah: float
    satisfies_soc_constraint: bool


def _extract_columns(
    data: pd.DataFrame,
    time_col: str = "time_s",
    solar_col: str = "P_solar",
    prop_col: str = "P_prop",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if {time_col, solar_col, prop_col}.issubset(data.columns):
        time_s = data[time_col].to_numpy(dtype=float)
        p_solar = data[solar_col].to_numpy(dtype=float)
        p_prop = data[prop_col].to_numpy(dtype=float)
    else:
        if data.shape[1] < 3:
            raise ValueError("STK input must contain at least three columns")
        time_s = data.iloc[:, 0].to_numpy(dtype=float)
        p_solar = data.iloc[:, 1].to_numpy(dtype=float)
        p_prop = data.iloc[:, 2].to_numpy(dtype=float)
    if len(time_s) < 2:
        raise ValueError("At least two time points are required")
    time_s = time_s - time_s[0]
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("Time stamps must be strictly increasing")
    return time_s, p_solar, p_prop


def _battery_current_from_power(
    p_solar: np.ndarray,
    p_prop: np.ndarray,
    p_load: float,
    bus_voltage: float,
) -> tuple[np.ndarray, np.ndarray]:
    p_net = p_solar - p_load - p_prop
    current_battery = -p_net / bus_voltage
    return p_net, current_battery


def _cumulative_discharge_ah(time_s: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    delta_t_h = np.diff(time_s) / 3600.0
    average_current = 0.5 * (current_a[1:] + current_a[:-1])
    cumulative = np.concatenate(([0.0], np.cumsum(average_current * delta_t_h)))
    return cumulative


def _estimate_capacity_from_coulomb_counting(
    time_s: np.ndarray, current_a: np.ndarray, soc_lower_bound: float
) -> tuple[float, float]:
    cumulative = _cumulative_discharge_ah(time_s, current_a)
    cumulative_floor = np.minimum.accumulate(cumulative)
    discharge_depth = cumulative - cumulative_floor
    max_discharge_depth = float(np.max(discharge_depth))
    usable_soc_window = 1.0 - soc_lower_bound
    if usable_soc_window <= 0:
        raise ValueError("soc_lower_bound must be < 1")
    theoretical_capacity = max_discharge_depth / usable_soc_window
    return max(theoretical_capacity, 1e-8), max_discharge_depth


def _verify_with_pybamm(
    time_s: np.ndarray, current_a: np.ndarray, capacity_ah: float
) -> float:
    clipped_current = _clip_charge_at_full_soc(time_s, current_a, capacity_ah)
    model = pybamm.equivalent_circuit.Thevenin()
    parameter_values = model.default_parameter_values.copy()
    parameter_values.update(
        {
            "Cell capacity [A.h]": capacity_ah,
            "Nominal cell capacity [A.h]": capacity_ah,
            "Initial SoC": 0.999,
            "Current function [A]": pybamm.Interpolant(time_s, clipped_current, pybamm.t),
        }
    )
    simulation = pybamm.Simulation(model, parameter_values=parameter_values)
    solution = simulation.solve(t_eval=time_s)
    return float(np.min(solution["SoC"].entries))


def _clip_charge_at_full_soc(
    time_s: np.ndarray, current_a: np.ndarray, capacity_ah: float
) -> np.ndarray:
    clipped = current_a.copy()
    soc = 0.999
    for k in range(len(time_s) - 1):
        dt_h = (time_s[k + 1] - time_s[k]) / 3600.0
        trial_soc = soc - clipped[k] * dt_h / capacity_ah
        if trial_soc > 0.999999:
            min_current = -(0.999999 - soc) * capacity_ah / max(dt_h, 1e-12)
            clipped[k] = max(clipped[k], min_current)
            trial_soc = 0.999999
        soc = trial_soc
    if len(clipped) > 1:
        clipped[-1] = clipped[-2]
    return clipped


def optimize_minimum_capacity(
    stk_data: pd.DataFrame,
    p_load: float = 3000.0,
    bus_voltage: float = 100.0,
    soc_lower_bound: float = 0.7,
) -> OptimizationResult:
    time_s, p_solar, p_prop = _extract_columns(stk_data)
    _, current_battery = _battery_current_from_power(
        p_solar, p_prop, p_load=p_load, bus_voltage=bus_voltage
    )
    theoretical_capacity_ah, _ = _estimate_capacity_from_coulomb_counting(
        time_s, current_battery, soc_lower_bound=soc_lower_bound
    )
    required_capacity, minimum_soc, satisfies = _find_verified_capacity(
        time_s=time_s,
        current_a=current_battery,
        soc_lower_bound=soc_lower_bound,
        initial_capacity=theoretical_capacity_ah,
    )
    return OptimizationResult(
        theoretical_capacity_ah=theoretical_capacity_ah,
        verified_capacity_ah=required_capacity,
        minimum_soc=minimum_soc,
        capacity_margin_ah=max(required_capacity - theoretical_capacity_ah, 0.0),
        satisfies_soc_constraint=satisfies,
    )


def _find_verified_capacity(
    time_s: np.ndarray,
    current_a: np.ndarray,
    soc_lower_bound: float,
    initial_capacity: float,
) -> tuple[float, float, bool]:
    cache: dict[float, float] = {}

    def min_soc(capacity_ah: float) -> float:
        key = round(capacity_ah, 10)
        if key not in cache:
            cache[key] = _verify_with_pybamm(time_s, current_a, capacity_ah)
        return cache[key]

    high = max(initial_capacity, 1e-6)
    high_soc = min_soc(high)
    for _ in range(20):
        if high_soc >= soc_lower_bound:
            break
        high *= 1.2
        high_soc = min_soc(high)
    else:
        return high, high_soc, False

    low = high / 2
    low_soc = min_soc(low)
    for _ in range(20):
        if low_soc < soc_lower_bound:
            break
        high = low
        high_soc = low_soc
        low /= 2
        low_soc = min_soc(low)

    if low_soc >= soc_lower_bound:
        return low, low_soc, True

    for _ in range(25):
        mid = 0.5 * (low + high)
        mid_soc = min_soc(mid)
        if mid_soc >= soc_lower_bound:
            high = mid
            high_soc = mid_soc
        else:
            low = mid
    return high, high_soc, True


def synthetic_stk_data() -> pd.DataFrame:
    dt = 60.0
    orbit_period = 5400.0
    cycles = 6
    time_s = np.arange(0.0, cycles * orbit_period + dt, dt)
    phase = np.mod(time_s, orbit_period)
    eclipse = (phase > 2200.0) & (phase < 3600.0)
    p_solar = np.where(eclipse, 0.0, 4500.0)
    p_prop = np.where((phase > 1200.0) & (phase < 1800.0), 1800.0, 0.0)
    p_prop = np.where((phase > 4200.0) & (phase < 4500.0), 900.0, p_prop)
    return pd.DataFrame({"time_s": time_s, "P_solar": p_solar, "P_prop": p_prop})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Low-earth-orbit satellite battery and electric propulsion co-optimization"
    )
    parser.add_argument(
        "--stk-csv",
        type=str,
        default=None,
        help="Path to STK exported CSV file. If omitted, a synthetic profile is used.",
    )
    parser.add_argument("--p-load", type=float, default=3000.0, help="Constant load power [W]")
    parser.add_argument("--bus-voltage", type=float, default=100.0, help="Bus voltage [V]")
    parser.add_argument("--soc-min", type=float, default=0.7, help="Minimum allowed SOC")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stk_csv is None:
        stk_data = synthetic_stk_data()
    else:
        stk_data = pd.read_csv(args.stk_csv)
    result = optimize_minimum_capacity(
        stk_data=stk_data,
        p_load=args.p_load,
        bus_voltage=args.bus_voltage,
        soc_lower_bound=args.soc_min,
    )
    print(f"理论最小标称容量 C_min: {result.theoretical_capacity_ah:.4f} Ah")
    print(f"PyBaMM 校核后最小容量: {result.verified_capacity_ah:.4f} Ah")
    print(f"PyBaMM 验证最小 SOC: {result.minimum_soc * 100:.2f}%")
    print(f"为满足 SOC 约束建议附加容量: {result.capacity_margin_ah:.4f} Ah")
    print(f"SOC 约束满足: {result.satisfies_soc_constraint}")


if __name__ == "__main__":
    main()
