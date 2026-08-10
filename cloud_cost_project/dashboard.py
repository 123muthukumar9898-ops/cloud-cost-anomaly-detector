import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cloud Cost Anomaly Detector", layout="wide")

st.title("☁️ AI-Powered Cloud Cost Anomaly Detector")
st.markdown("Detects unusual spikes in daily cloud spending using a Z-score based anomaly detection model.")

df = pd.read_csv("cloud_billing_data_with_predictions.csv")
df["date"] = pd.to_datetime(df["date"])

st.sidebar.header("Filters")
services = ["All"] + sorted(df["service"].unique().tolist())
selected_service = st.sidebar.selectbox("Select Service", services)

if selected_service != "All":
    filtered_df = df[df["service"] == selected_service]
else:
    filtered_df = df

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(filtered_df))
col2.metric("Anomalies Detected", int(filtered_df["predicted_anomaly"].sum()))
col3.metric("Total Cost ($)", f"{filtered_df['daily_cost'].sum():,.2f}")

st.subheader(f"Daily Cost Trend — {selected_service}")

fig, ax = plt.subplots(figsize=(12, 5))

if selected_service == "All":
    for service in df["service"].unique():
        service_df = filtered_df[filtered_df["service"] == service].sort_values("date")
        ax.plot(service_df["date"], service_df["daily_cost"], label=service, alpha=0.6)
else:
    service_df = filtered_df.sort_values("date")
    ax.plot(service_df["date"], service_df["daily_cost"], label=selected_service, color="steelblue")

anomaly_points = filtered_df[filtered_df["predicted_anomaly"] == 1]
ax.scatter(anomaly_points["date"], anomaly_points["daily_cost"], color="red", s=60, label="Anomaly", zorder=5)

ax.set_xlabel("Date")
ax.set_ylabel("Cost ($)")
ax.legend()
ax.grid(alpha=0.3)
plt.xticks(rotation=45)

st.pyplot(fig)

st.subheader("⚠️ Detected Anomalies")
anomaly_table = filtered_df[filtered_df["predicted_anomaly"] == 1][
    ["date", "service", "daily_cost", "z_score"]
].sort_values("date", ascending=False)
st.dataframe(anomaly_table, use_container_width=True)