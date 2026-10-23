"""Task 4 - Streaming Inference: 10 unseen records, one at a time, 1 s delay."""
import time

import joblib
import pandas as pd

from task1_train_model import FEATURES, TARGET

model = joblib.load("model.pkl")
unseen = pd.read_csv("data/unseen_data.csv")
stream = unseen.iloc[21:31]          # 10 unseen records

print("=== STREAMING INFERENCE (1 record / second) ===")
for n, (_, row) in enumerate(stream.iterrows(), start=1):
    time.sleep(1)                    # simulate arrival of a new record
    record = row[FEATURES].to_frame().T
    pred = int(model.predict(record)[0])
    prob = float(model.predict_proba(record)[0][1])
    print(f"[{time.strftime('%H:%M:%S')}] Record {n:>2} | "
          f"Pclass={row['Pclass']} Sex={row['Sex']:<6} Age={row['Age']} "
          f"Fare={row['Fare']:.2f} -> "
          f"{'Survived' if pred else 'Did not survive':<15} "
          f"(p={prob:.2f}, actual={int(row[TARGET])})", flush=True)
print("Stream finished.")
