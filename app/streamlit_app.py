"""Phase 1 interface placeholder; no jobs or fabricated study metrics."""

from pathlib import Path

import streamlit as st

from shiftlab.experiments.config import ConfigError, load_config
from shiftlab.scope import FINAL_OBJECTIVE

st.set_page_config(page_title="Domain Shift Workbench", page_icon="🔬", layout="wide")
st.title("Domain Shift Workbench")
st.caption("Phase 1 · Research foundation")
st.write(FINAL_OBJECTIVE)
st.info("No completed experiments yet. Dataset preparation begins in Phase 2.")
st.write(
    "This foundation can validate a configuration. Training, results, diagnostics "
    "and report exports are implemented in later phases."
)

name = st.selectbox("Configuration", ("study", "smoke"))
path = Path(__file__).resolve().parents[1] / "configs" / f"{name}.yaml"
try:
    config = load_config(path)
except ConfigError as error:
    st.error(f"Configuration cannot be validated: {error}")
else:
    st.success(f"Configuration valid: {config.study_name}")
    st.write("Default stage: fit only. Validation does not prepare data or train a model.")
    with st.expander("Validated configuration"):
        st.json(config.model_dump(mode="json"))

st.write("Run the offline environment check from the project terminal:")
st.code("shiftlab doctor", language="text")
