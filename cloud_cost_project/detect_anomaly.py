import pandas as pd
import numpy as np

df = pd.read_csv("cloud_billing_data.csv")
print(f"Loaded {len(df)} records.\n")

service_stats = df.groupby("service")["daily_cost"].agg(["mean", "std"]).reset_index()
service_stats.columns = ["service", "mean_cost", "std_cost"]
print("Service-wise average cost and variation:")
print(service_stats)
print()

df = df.merge(service_stats, on="service")

df["z_score"] = (df["daily_cost"] - df["mean_cost"]) / df["std_cost"]

THRESHOLD = 3
df["predicted_anomaly"] = (df["z_score"].abs() > THRESHOLD).astype(int)

correct = (df["predicted_anomaly"] == df["is_anomaly"]).sum()
total = len(df)
accuracy = correct / total * 100

print(f"Total anomalies detected: {df['predicted_anomaly'].sum()}")
print(f"Actual anomalies (ground truth): {df['is_anomaly'].sum()}")
print(f"Accuracy: {accuracy:.2f}%\n")

anomalies = df[df["predicted_anomaly"] == 1][["date", "service", "daily_cost", "z_score"]]
print("Detected anomalies:")
print(anomalies.to_string(index=False))

df.to_csv("cloud_billing_data_with_predictions.csv", index=False)
print("\nResults saved to cloud_billing_data_with_predictions.csv")