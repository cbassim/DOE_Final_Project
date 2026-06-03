from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# --------------------------------------------------
# Factor definitions in real units
# --------------------------------------------------
# Student-facing ranges are in real units. Internally, the hidden model still
# works in coded units on approximately [-1, +1].

LED_MIN, LED_CENTER, LED_MAX = 6.0, 14.0, 22.0
PRESSURE_MIN, PRESSURE_CENTER, PRESSURE_MAX = 4.0, 13.0, 22.0
FILTER_MIN, FILTER_CENTER, FILTER_MAX = 1.0, 9.0, 17.0

LED_HALF_RANGE = (LED_MAX - LED_MIN) / 2.0
PRESSURE_HALF_RANGE = (PRESSURE_MAX - PRESSURE_MIN) / 2.0
FILTER_HALF_RANGE = (FILTER_MAX - FILTER_MIN) / 2.0

BUDGET = 35


@dataclass(frozen=True)
class HiddenModel:
    family_index: int
    family_name: str
    beta0: float
    betaA: float
    betaB: float
    betaC: float
    betaAB: float
    betaAC: float
    betaBC: float
    qA: float
    qB: float
    qC: float
    sigma: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------
# Seed helpers
# --------------------------------------------------

def make_seed(identifier: str) -> int:
    """Convert any text identifier into a reproducible 32-bit integer seed."""
    clean = identifier.strip().lower()
    digest = hashlib.sha256(clean.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


# --------------------------------------------------
# Coding helpers
# --------------------------------------------------

def validate_real_inputs(led: float, pressure: float, filt: float) -> None:
    if not (LED_MIN <= led <= LED_MAX):
        raise ValueError(f"LED Current must be between {LED_MIN:g} and {LED_MAX:g} mA.")
    if not (PRESSURE_MIN <= pressure <= PRESSURE_MAX):
        raise ValueError(
            f"Strap Pressure must be between {PRESSURE_MIN:g} and {PRESSURE_MAX:g} kPa."
        )
    if not (FILTER_MIN <= filt <= FILTER_MAX):
        raise ValueError(
            f"Filter Window Length must be between {FILTER_MIN:g} and {FILTER_MAX:g} samples."
        )


def real_to_coded(led: float, pressure: float, filt: float) -> tuple[float, float, float]:
    validate_real_inputs(led, pressure, filt)
    xA = (led - LED_CENTER) / LED_HALF_RANGE
    xB = (pressure - PRESSURE_CENTER) / PRESSURE_HALF_RANGE
    xC = (filt - FILTER_CENTER) / FILTER_HALF_RANGE
    return xA, xB, xC


def coded_to_real(xA: float, xB: float, xC: float) -> tuple[float, float, float]:
    led = LED_CENTER + xA * LED_HALF_RANGE
    pressure = PRESSURE_CENTER + xB * PRESSURE_HALF_RANGE
    filt = FILTER_CENTER + xC * FILTER_HALF_RANGE
    return led, pressure, filt


# --------------------------------------------------
# Hidden surface family specification
# --------------------------------------------------

def _sample_family_parameters(
    rng: np.random.Generator, family_index: int
) -> dict[str, float | int | str]:
    """
    Generate one parameter set from a controlled family of teachable surfaces.

    Design intent for this final version:
    - A and B should dominate broad screening.
    - Filter should be present but weak.
    - The first local A/B box should naturally point to the upper-right corner.
    - An aggressive push farther toward the upper-right boundary should usually
      overshoot, making re-boxing and a second-order follow-up feel natural.
    """
    if family_index == 0:
        # Fairly balanced A/B dome, optimum clearly interior.
        return {
            "family_index": 0,
            "family_name": "centered_broad",
            "xA_star": float(rng.uniform(0.50, 0.64)),
            "xB_star": float(rng.uniform(0.54, 0.66)),
            "xC_star": float(rng.uniform(-0.02, 0.02)),
            "qA": float(rng.uniform(7.8, 9.0)),
            "qB": float(rng.uniform(8.8, 10.0)),
            "qC": float(rng.uniform(0.26, 0.40)),
            "betaAB": float(rng.uniform(-1.0, 1.0)),
            "betaAC": float(rng.uniform(-0.02, 0.02)),
            "betaBC": float(rng.uniform(-0.02, 0.02)),
            "target_peak": float(rng.uniform(66.0, 70.0)),
            "sigma": float(rng.uniform(0.65, 0.95)),
        }
    if family_index == 1:
        # Rotated A/B contours, still with an interior optimum and weak filter.
        ab_mag = float(rng.uniform(2.4, 3.5))
        ab_sign = -1.0 if rng.random() < 0.5 else 1.0
        return {
            "family_index": 1,
            "family_name": "rotated_northeast",
            "xA_star": float(rng.uniform(0.52, 0.68)),
            "xB_star": float(rng.uniform(0.56, 0.70)),
            "xC_star": float(rng.uniform(-0.02, 0.02)),
            "qA": float(rng.uniform(8.0, 9.2)),
            "qB": float(rng.uniform(8.6, 9.8)),
            "qC": float(rng.uniform(0.26, 0.40)),
            "betaAB": float(ab_sign * ab_mag),
            "betaAC": float(rng.uniform(-0.02, 0.02)),
            "betaBC": float(rng.uniform(-0.02, 0.02)),
            "target_peak": float(rng.uniform(67.0, 72.0)),
            "sigma": float(rng.uniform(0.65, 0.95)),
        }
    if family_index == 2:
        # LED matters a bit more than Pressure, but Pressure still belongs in Phase II.
        return {
            "family_index": 2,
            "family_name": "led_skewed",
            "xA_star": float(rng.uniform(0.62, 0.80)),
            "xB_star": float(rng.uniform(0.50, 0.62)),
            "xC_star": float(rng.uniform(-0.02, 0.02)),
            "qA": float(rng.uniform(8.8, 10.0)),
            "qB": float(rng.uniform(7.8, 9.0)),
            "qC": float(rng.uniform(0.26, 0.40)),
            "betaAB": float(rng.uniform(-1.8, 1.2)),
            "betaAC": float(rng.uniform(-0.02, 0.02)),
            "betaBC": float(rng.uniform(-0.02, 0.02)),
            "target_peak": float(rng.uniform(66.0, 71.0)),
            "sigma": float(rng.uniform(0.65, 1.00)),
        }
    # Pressure matters a bit more than LED, but both still drive the story.
    return {
        "family_index": 3,
        "family_name": "pressure_skewed",
        "xA_star": float(rng.uniform(0.42, 0.56)),
        "xB_star": float(rng.uniform(0.60, 0.72)),
        "xC_star": float(rng.uniform(-0.02, 0.02)),
        "qA": float(rng.uniform(7.8, 9.0)),
        "qB": float(rng.uniform(8.8, 10.2)),
        "qC": float(rng.uniform(0.26, 0.40)),
        "betaAB": float(rng.uniform(-1.2, 1.8)),
        "betaAC": float(rng.uniform(-0.02, 0.02)),
        "betaBC": float(rng.uniform(-0.02, 0.02)),
        "target_peak": float(rng.uniform(66.0, 71.0)),
        "sigma": float(rng.uniform(0.65, 1.00)),
    }


def _build_candidate_from_params(params: dict[str, float | int | str]) -> HiddenModel:
    """Construct the quadratic model so the chosen coded optimum is built in."""
    family_index = int(params["family_index"])
    family_name = str(params["family_name"])

    xA_star = float(params["xA_star"])
    xB_star = float(params["xB_star"])
    xC_star = float(params["xC_star"])
    qA = float(params["qA"])
    qB = float(params["qB"])
    qC = float(params["qC"])
    betaAB = float(params["betaAB"])
    betaAC = float(params["betaAC"])
    betaBC = float(params["betaBC"])
    target_peak = float(params["target_peak"])
    sigma = float(params["sigma"])

    # Stationary conditions at the target optimum.
    betaA = 2.0 * qA * xA_star - betaAB * xB_star - betaAC * xC_star
    betaB = 2.0 * qB * xB_star - betaAB * xA_star - betaBC * xC_star
    betaC = 2.0 * qC * xC_star - betaAC * xA_star - betaBC * xB_star

    # Back-calculate beta0 so the peak height is where we want it.
    beta0 = target_peak - (
        betaA * xA_star
        + betaB * xB_star
        + betaC * xC_star
        + betaAB * xA_star * xB_star
        + betaAC * xA_star * xC_star
        + betaBC * xB_star * xC_star
        - qA * xA_star**2
        - qB * xB_star**2
        - qC * xC_star**2
    )

    return HiddenModel(
        family_index=family_index,
        family_name=family_name,
        beta0=float(beta0),
        betaA=float(betaA),
        betaB=float(betaB),
        betaC=float(betaC),
        betaAB=float(betaAB),
        betaAC=float(betaAC),
        betaBC=float(betaBC),
        qA=float(qA),
        qB=float(qB),
        qC=float(qC),
        sigma=float(sigma),
    )


# --------------------------------------------------
# Model evaluation and checks
# --------------------------------------------------

def predict_noiseless_score(model: HiddenModel, xA: float, xB: float, xC: float) -> float:
    return (
        model.beta0
        + model.betaA * xA
        + model.betaB * xB
        + model.betaC * xC
        + model.betaAB * xA * xB
        + model.betaAC * xA * xC
        + model.betaBC * xB * xC
        - model.qA * xA**2
        - model.qB * xB**2
        - model.qC * xC**2
    )


def _ab_stationary_point(model: HiddenModel) -> tuple[float, float] | None:
    """Stationary point in the A/B plane with xC fixed at 0."""
    M = np.array(
        [[2.0 * model.qA, -model.betaAB], [-model.betaAB, 2.0 * model.qB]],
        dtype=float,
    )
    rhs = np.array([model.betaA, model.betaB], dtype=float)
    try:
        x_star = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return None
    return float(x_star[0]), float(x_star[1])


def _screening_main_effects(model: HiddenModel) -> dict[str, float]:
    """Noiseless 2^3 screening main effects in coded units at ±1 levels."""
    rows: list[tuple[float, float, float, float]] = []
    for xA in (-1.0, 1.0):
        for xB in (-1.0, 1.0):
            for xC in (-1.0, 1.0):
                y = predict_noiseless_score(model, xA, xB, xC)
                rows.append((xA, xB, xC, y))

    def main_effect(index: int) -> float:
        highs = [y for row in rows if row[index] > 0 for y in [row[3]]]
        lows = [y for row in rows if row[index] < 0 for y in [row[3]]]
        return float(np.mean(highs) - np.mean(lows))

    return {"A": main_effect(0), "B": main_effect(1), "C": main_effect(2)}


def _broad_curvature_gap(model: HiddenModel) -> float:
    """Center-point signal relative to the average of the full 2^3 corners."""
    corners = [
        predict_noiseless_score(model, xA, xB, xC)
        for xA in (-1.0, 1.0)
        for xB in (-1.0, 1.0)
        for xC in (-1.0, 1.0)
    ]
    return float(predict_noiseless_score(model, 0.0, 0.0, 0.0) - np.mean(corners))


def _score_real(model: HiddenModel, led: float, pressure: float, filt: float = FILTER_CENTER) -> float:
    xA, xB, xC = real_to_coded(led, pressure, filt)
    return predict_noiseless_score(model, xA, xB, xC)


def model_passes_checks(model: HiddenModel) -> bool:
    # Keep a single clear optimum in A/B.
    h_det = 4.0 * model.qA * model.qB - model.betaAB**2
    if h_det <= 0.0:
        return False

    stationary = _ab_stationary_point(model)
    if stationary is None:
        return False
    xA_star, xB_star = stationary

    # Interior enough that students can move toward the optimum, then overstep it.
    if not (0.40 <= xA_star <= 0.82):
        return False
    if not (0.52 <= xB_star <= 0.82):
        return False

    center_score = predict_noiseless_score(model, 0.0, 0.0, 0.0)
    peak_score = predict_noiseless_score(model, xA_star, xB_star, 0.0)

    if not (58.0 <= center_score <= 67.0):
        return False
    if not (65.0 <= peak_score <= 74.0):
        return False
    if not (5.0 <= peak_score - center_score <= 11.0):
        return False

    # Broad screening should suggest curvature.
    if _broad_curvature_gap(model) < 15.0:
        return False

    # Filter should stay weak enough that A and B dominate screening.
    screening_effects = _screening_main_effects(model)
    if abs(screening_effects["A"]) < 7.0:
        return False
    if abs(screening_effects["B"]) < 7.0:
        return False
    if abs(screening_effects["C"]) > 0.25:
        return False

    # First local 2^2 box around (16,16) should naturally point to the upper-right corner.
    y_14_14 = _score_real(model, 14.0, 14.0)
    y_18_14 = _score_real(model, 18.0, 14.0)
    y_14_18 = _score_real(model, 14.0, 18.0)
    y_18_18 = _score_real(model, 18.0, 18.0)
    if y_18_18 < max(y_14_14, y_18_14, y_14_18) + 0.4:
        return False

    # A slightly more interior re-box center should improve a bit.
    y_rebox_center = _score_real(model, 18.5, 19.5)
    if y_rebox_center < y_18_18 + 0.05:
        return False

    # An aggressive move farther up and right should overstep.
    y_overshoot = _score_real(model, 20.0, 22.0)
    if y_overshoot > y_18_18 - 1.0:
        return False

    return True


def generate_hidden_model(student_id: str, max_tries: int = 8000) -> HiddenModel:
    """Generate a reproducible hidden model and guarantee it passes validation."""
    base_seed = make_seed(student_id)
    family_index = base_seed % 4

    for k in range(max_tries):
        rng = np.random.default_rng(base_seed + 1009 * k + 100_000 * family_index)
        params = _sample_family_parameters(rng, family_index)
        candidate = _build_candidate_from_params(params)
        if model_passes_checks(candidate):
            return candidate

    raise RuntimeError(
        "Could not generate a valid hidden model. "
        "This should be rare. Try a different ID or increase max_tries."
    )


# --------------------------------------------------
# Observation helpers
# --------------------------------------------------

def make_run_rng(student_id: str, run_number: int) -> np.random.Generator:
    seed = make_seed(f"{student_id}_run_{run_number}")
    return np.random.default_rng(seed)


def observe_score(
    model: HiddenModel,
    led: float,
    pressure: float,
    filt: float,
    rng: np.random.Generator,
    *,
    include_true_score: bool = False,
) -> dict[str, Any]:
    xA, xB, xC = real_to_coded(led, pressure, filt)
    y_true = predict_noiseless_score(model, xA, xB, xC)
    y_obs = y_true + float(rng.normal(0.0, model.sigma))
    y_obs = float(np.clip(y_obs, 0.0, 100.0))

    result: dict[str, Any] = {
        "LED_Current_mA": float(led),
        "Strap_Pressure_kPa": float(pressure),
        "Filter_Window_samples": float(filt),
        "A_coded": float(xA),
        "B_coded": float(xB),
        "C_coded": float(xC),
        "PPG_Quality_Score": round(y_obs, 2),
    }
    if include_true_score:
        result["PPG_Quality_True"] = round(float(y_true), 2)
    return result


def run_experiment(
    student_id: str,
    run_number: int,
    led: float,
    pressure: float,
    filt: float,
    model: HiddenModel,
    *,
    include_true_score: bool = False,
) -> dict[str, Any]:
    run_rng = make_run_rng(student_id, run_number)
    result = observe_score(
        model,
        led=led,
        pressure=pressure,
        filt=filt,
        rng=run_rng,
        include_true_score=include_true_score,
    )
    result["run_number"] = int(run_number)
    return result


# --------------------------------------------------
# Optional instructor helpers
# --------------------------------------------------

def model_summary(model: HiddenModel) -> dict[str, float | tuple[float, float] | None | str | int]:
    stationary_ab = _ab_stationary_point(model)
    stationary_real = None
    if stationary_ab is not None:
        led_star, pressure_star, _ = coded_to_real(stationary_ab[0], stationary_ab[1], 0.0)
        stationary_real = (round(led_star, 3), round(pressure_star, 3))

    y_18_18 = _score_real(model, 18.0, 18.0)
    y_rebox_center = _score_real(model, 18.5, 19.5)
    y_overshoot = _score_real(model, 20.0, 22.0)

    return {
        **model.to_dict(),
        "AB_stationary_coded": stationary_ab,
        "AB_stationary_real": stationary_real,
        "center_score_true": round(predict_noiseless_score(model, 0.0, 0.0, 0.0), 3),
        "high_high_score_true": round(predict_noiseless_score(model, 1.0, 1.0, 0.0), 3),
        "broad_curvature_gap_true": round(_broad_curvature_gap(model), 3),
        "local_upper_right_true": round(y_18_18, 3),
        "suggested_rebox_center_true": round(y_rebox_center, 3),
        "aggressive_overshoot_true": round(y_overshoot, 3),
    }
