import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

NUM_DAYS = 180
SERVICES = ["EC2", "S3", "Lambda", "RDS", "CloudFront"]
START_DATE = datetime.today() - timedelta(days=NUM_DAYS)

SERVICE_BASE_COST = {
    "EC2": (40, 5),
    "S3": (8, 1.5),
    "Lambda": (5, 1),
    "RDS": (20, 3),
    "CloudFront": (10, 2),
}

records = []

for service, (mean_cost, std_cost) in SERVICE_BASE_COST.items():
    for day_offset in range(NUM_DAYS):
        date = START_DATE + timedelta(days=day_offset)
        weekday_factor = 1.15 if date.weekday() < 5 else 0.85
        cost = np.random.normal(mean_cost, std_cost) * weekday_factor
        cost = max(cost, 0.5)

        is_anomaly = 0
        if np.random.rand() < 0.03:
            spike_multiplier = np.random.uniform(3, 8)
            cost = cost * spike_multiplier
            is_anomaly = 1

        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "service": service,
            "daily_cost": round(cost, 2),
            "is_anomaly": is_anomaly
        })

df = pd.DataFrame(records)
df = df.sort_values(["date", "service"]).reset_index(drop=True)

output_path = "cloud_billing_data.csv"
df.to_csv(output_path, index=False)

print(f"Generated {len(df)} records across {NUM_DAYS} days for {len(SERVICE_BASE_COST)} services.")
print(f"Total anomalies injected: {df['is_anomaly'].sum()}")
print(f"Saved to: {output_path}")
print("\nSample rows:")
print(df.head(10))