"""Task 3 - Batch Inference: predict 20 unseen records together, save CSV."""
import joblib
import pandas as pd

from task1_train_model import FEATURES, TARGET

model = joblib.load("model.pkl")
unseen = pd.read_csv("data/unseen_data.csv")

batch = unseen.iloc[1:21].copy()     # 20 unseen records (different from Task 2's)
batch["Predicted_Survived"] = model.predict(batch[FEATURES])      # one call
batch["Survival_Probability"] = model.predict_proba(batch[FEATURES])[:, 1].round(4)
batch = batch.rename(columns={TARGET: "Actual_Survived"})

batch.to_csv("batch_predictions.csv", index=False)
print("=== BATCH INFERENCE (20 records) ===")
print(batch[["RecordId", "Pclass", "Sex", "Age", "Fare",
             "Predicted_Survived", "Survival_Probability",
             "Actual_Survived"]].to_string(index=False))
print(f"\nSaved {len(batch)} predictions to batch_predictions.csv")
