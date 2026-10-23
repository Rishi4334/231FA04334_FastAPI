"""Task 2 - Online Inference: predict ONE unseen passenger using saved model.pkl."""
import joblib
import pandas as pd

from task1_train_model import FEATURES, TARGET

model = joblib.load("model.pkl")
unseen = pd.read_csv("data/unseen_data.csv")

record = unseen.iloc[[0]]            # one unseen passenger
pred = int(model.predict(record[FEATURES])[0])
prob = float(model.predict_proba(record[FEATURES])[0][1])

print("=== ONLINE INFERENCE (single record) ===")
print("Passenger details:")
print(record[FEATURES].iloc[0].to_string())
print(f"\nPredicted: {'Survived' if pred else 'Did not survive'} "
      f"(class={pred}, survival probability={prob:.2%})")
print(f"Actual   : {'Survived' if record[TARGET].iloc[0] else 'Did not survive'}")
