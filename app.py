from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from hidden_system import (
    BUDGET,
    FILTER_MAX,
    FILTER_MIN,
    LED_MAX,
    LED_MIN,
    PRESSURE_MAX,
    PRESSURE_MIN,
    generate_hidden_model,
    model_summary,
    run_experiment,
    validate_real_inputs,
)

SHOW_INSTRUCTOR_PANEL = False

# Optional branding files placed beside app.py
APP_DIR = Path(__file__).resolve().parent
icon_path = APP_DIR / "ibio_icon.png"
logo_path = APP_DIR / "ibio_logo.png"
page_icon: str = "ibio_logo.png"
if icon_path.exists():
    page_icon = str(icon_path)

st.set_page_config(
    page_title="PPG DOE Optimization Lab",
    page_icon=page_icon,
    layout="wide",
)

if logo_path.exists():
    if icon_path.exists():
        st.logo(str(logo_path), icon_image=str(icon_path))
    else:
        st.logo(str(logo_path))


# --------------------------------------------------
# Session helpers
# --------------------------------------------------

def _init_session() -> None:
    defaults: dict[str, Any] = {
        "student_id": "",
        "model": None,
        "history": [],
        "session_started": False,
        "last_result": None,
        "last_duplicate_run": None,
        "led_input": "",
        "pressure_input": "",
        "filter_input": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_session() -> None:
    st.session_state["model"] = None
    st.session_state["history"] = []
    st.session_state["session_started"] = False
    st.session_state["last_result"] = None
    st.session_state["last_duplicate_run"] = None
    st.session_state["led_input"] = ""
    st.session_state["pressure_input"] = ""
    st.session_state["filter_input"] = ""


def _normalize_id(raw_value: str) -> str:
    return raw_value.strip()


def _find_duplicate_run(
    history: list[dict[str, Any]],
    led: float,
    pressure: float,
    filt: float,
) -> int | None:
    for prior in history:
        if (
            prior["LED_Current_mA"] == float(led)
            and prior["Strap_Pressure_kPa"] == float(pressure)
            and prior["Filter_Window_samples"] == float(filt)
        ):
            return int(prior["run_number"])
    return None


_init_session()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("Session")
    entered_id = st.text_input(
        "Student ID",
        value=st.session_state["student_id"],
        help="Use the same ID every time so your hidden system stays consistent.",
    )

    col_start, col_reset = st.columns(2)
    with col_start:
        if st.button("Start / Refresh", use_container_width=True):
            clean_id = _normalize_id(entered_id)
            if not clean_id:
                st.error("Enter a Student or Group ID first.")
            else:
                current_id = st.session_state["student_id"]
                id_changed = bool(current_id and clean_id != current_id)

                if id_changed:
                    _reset_session()
                    st.session_state["student_id"] = clean_id
                    st.session_state["model"] = generate_hidden_model(clean_id)
                    st.session_state["session_started"] = True
                    st.info("New ID detected. Previous run history was cleared before starting the new session.")
                else:
                    st.session_state["student_id"] = clean_id
                    if st.session_state["model"] is None:
                        st.session_state["model"] = generate_hidden_model(clean_id)
                    st.session_state["session_started"] = True
                    if st.session_state["history"]:
                        st.success("Session refreshed. Existing run history for this ID was kept.")
                    else:
                        st.success("Session ready.")
    with col_reset:
        if st.button("Clear Runs", use_container_width=True):
            _reset_session()
            st.session_state["student_id"] = _normalize_id(entered_id)
            st.info("Run history cleared.")

    runs_used = len(st.session_state["history"])
    st.metric("Runs used", runs_used)
    st.metric("Runs remaining", max(BUDGET - runs_used, 0))

    st.markdown("---")
    st.caption(
        "Work only in real-world units here."
    )


# --------------------------------------------------
# Main page
# --------------------------------------------------

st.title("Wearable PPG Sensor Optimization")

st.markdown(
    """
    In this virtual design-of-experiments lab, you are optimizing a wearable
    **photoplethysmography (PPG)** sensor. A PPG sensor estimates heart rate by
    shining light into the skin and measuring small changes in reflected light
    caused by blood volume changes.

    Under motion conditions, the signal can become noisy. Your job is to choose
    sensor settings that improve the **PPG Signal Quality Score** while working
    within a fixed experimental budget.

    You will control three factors:

    - **LED Current (mA):** the electrical current driving the optical LED. Higher
      current can increase the strength of the optical signal, but too much may
      introduce saturation or unnecessary power use.
    - **Strap Pressure (kPa):** how firmly the sensor is pressed against the skin.
      Too little pressure can allow motion artifacts, while too much pressure may
      reduce comfort or alter the local blood flow signal.
    - **Filter Window Length (samples):** the number of recent samples used to
      smooth the signal. A longer window can reduce noise, but may also make the
      sensor slower to respond to real changes.

    Your goal is to maximize **PPG Signal Quality Score** using no more than
    **35 experiments**. The best settings are not obvious in advance, so use your
    experiments thoughtfully.
    """
)

info1, info2, info3, info4 = st.columns(4)
info1.info(f"**LED Current:** {LED_MIN:g} to {LED_MAX:g} mA")
info2.info(f"**Strap Pressure:** {PRESSURE_MIN:g} to {PRESSURE_MAX:g} kPa")
info3.info(f"**Filter Window:** {FILTER_MIN:g} to {FILTER_MAX:g} samples")
info4.info(f"**Budget:** {BUDGET} runs")

st.markdown(
    "The underlying system is hidden. You will only observe experimental outcomes from the factor settings you choose. "
    "There is **no prescribed starting point** — deciding where to begin is part of the experiment."
)

if not st.session_state["session_started"] or st.session_state["model"] is None:
    st.warning("Enter the first letter of your last name and last 4 digits of your Student ID in the sidebar (as in B1234), then click **Start / Refresh** to begin.")

st.subheader("Run an Experiment")
st.caption("Enter your own starting settings in real units. Nothing is preloaded as the recommended start. Decimal settings are allowed for local follow-up and re-boxing.  Hit Run Experiement to submit the trial and observe an experimental response.")

with st.form("run_form", clear_on_submit=False, enter_to_submit=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        led_text = st.text_input(
            "LED Current (mA)",
            value=st.session_state["led_input"],
            placeholder=f"{LED_MIN:g} to {LED_MAX:g}",
        )
    with col2:
        pressure_text = st.text_input(
            "Strap Pressure (kPa)",
            value=st.session_state["pressure_input"],
            placeholder=f"{PRESSURE_MIN:g} to {PRESSURE_MAX:g}",
        )
    with col3:
        filt_text = st.text_input(
            "Filter Window Length (samples)",
            value=st.session_state["filter_input"],
            placeholder=f"{FILTER_MIN:g} to {FILTER_MAX:g}",
        )

    run_disabled = (
        not st.session_state["session_started"]
        or st.session_state["model"] is None
        or len(st.session_state["history"]) >= BUDGET
    )
    submitted = st.form_submit_button("Run Experiment", type="primary", disabled=run_disabled)

if submitted:
    try:
        if not led_text.strip() or not pressure_text.strip() or not filt_text.strip():
            raise ValueError("Enter all three factor settings before running an experiment.")

        led = float(led_text)
        pressure = float(pressure_text)
        filt = float(filt_text)
        validate_real_inputs(led=led, pressure=pressure, filt=filt)

        duplicate_run_number = _find_duplicate_run(
            st.session_state["history"],
            led=led,
            pressure=pressure,
            filt=filt,
        )

        next_run = len(st.session_state["history"]) + 1
        result = run_experiment(
            student_id=st.session_state["student_id"],
            run_number=next_run,
            led=led,
            pressure=pressure,
            filt=filt,
            model=st.session_state["model"],
        )
        st.session_state["history"].append(result)
        st.session_state["last_result"] = result
        st.session_state["last_duplicate_run"] = duplicate_run_number
        st.session_state["led_input"] = str(led_text)
        st.session_state["pressure_input"] = str(pressure_text)
        st.session_state["filter_input"] = str(filt_text)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not run experiment: {exc}")

last_result = st.session_state.get("last_result")
if last_result is not None:
    st.success(
        f"Run {last_result['run_number']} complete. "
        f"Observed PPG Signal Quality Score = {last_result['PPG_Quality_Score']:.2f}"
    )

last_duplicate_run = st.session_state.get("last_duplicate_run")
if last_duplicate_run is not None and last_result is not None:
    st.warning(
        f"These settings match your earlier Run {last_duplicate_run}. "
        "That can be useful if you intended a replicate."
    )

if len(st.session_state["history"]) >= BUDGET:
    st.warning("You have reached the 35-run budget. You can still review and download your data.")


# --------------------------------------------------
# Run history
# --------------------------------------------------

st.subheader("Run History")

history = st.session_state["history"]
if history:
    df = pd.DataFrame(history)

    student_df = df[
        [
            "run_number",
            "LED_Current_mA",
            "Strap_Pressure_kPa",
            "Filter_Window_samples",
            "PPG_Quality_Score",
        ]
    ].copy()

    st.dataframe(student_df, use_container_width=True, hide_index=True)

    csv_buffer = io.StringIO()
    student_df.to_csv(csv_buffer, index=False)

    st.download_button(
        "Download experimental data as CSV",
        data=csv_buffer.getvalue(),
        file_name=f"ppg_doe_runs_{st.session_state['student_id'] or 'student'}.csv",
        mime="text/csv",
    )
else:
    st.info("No experiments run yet.")


# --------------------------------------------------
# Notes area
# --------------------------------------------------

st.subheader("Experiment Notes")
st.text_area(
    "Optional notes",
    height=140,
    placeholder=(
        "Use this area for your own notes, such as:\n"
        "- chose a screening corner to start\n"
        "- filter held fixed for local search\n"
        "- moving toward higher LED and pressure\n"
        "- planning new local box"
    ),
)


# --------------------------------------------------
# Optional instructor/debug panel
# --------------------------------------------------

if SHOW_INSTRUCTOR_PANEL and st.session_state["model"] is not None:
    st.markdown("---")
    st.subheader("Instructor / Debug Panel")
    st.json(model_summary(st.session_state["model"]))
