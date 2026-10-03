import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="ML API Monitor", layout="wide")
st.title("ML API Monitor")
st.caption(f"Live view of {API_URL}")

try:
    response = requests.get(f"{API_URL}/monitoring", timeout=5)
    response.raise_for_status()
    data = response.json()
except requests.RequestException as exc:
    st.error(f"API unavailable: {exc}")
    st.stop()
except ValueError:
    st.error("API returned an invalid monitoring response.")
    st.stop()

metrics = st.columns(5)
metrics[0].metric("Predictions", data["successful_predictions"])
metrics[1].metric("Requests", data["total_requests"])
metrics[2].metric("Avg latency", f"{data['average_latency_ms']:.2f} ms" if data["average_latency_ms"] is not None else "n/a")
metrics[3].metric("P95 latency", f"{data['p95_latency_ms']:.2f} ms" if data["p95_latency_ms"] is not None else "n/a")
metrics[4].metric("Drift", data["drift"]["status"])

left, right = st.columns(2)
with left:
    st.subheader("Prediction distribution")
    st.bar_chart(pd.Series(data["prediction_distribution"], name="count"))
with right:
    st.subheader("Recent events")
    st.dataframe(pd.DataFrame(data["recent_events"]), use_container_width=True, hide_index=True)

st.subheader("Confidence")
confidence = [event["confidence"] for event in data["recent_events"]]
if confidence:
    st.line_chart(pd.DataFrame({"confidence": confidence}))
else:
    st.info("No prediction events recorded yet. Run the demo script.")

st.caption("Drift is a KS comparison of confidence windows, not proof of input-distribution shift.")
