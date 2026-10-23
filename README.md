# ML Model Deployment Strategies and Model Serving using FastAPI

## Project Description
This project trains a classification model on the Titanic dataset and demonstrates three inference
strategies (online, batch, streaming) plus REST API model serving with **FastAPI**.
The model is trained **once**; the saved pipeline (`model.pkl`, preprocessing + classifier) is reused for
every inference task and by the API.

## Dataset
Titanic - Machine Learning from Disaster (Kaggle): https://www.kaggle.com/competitions/titanic/data
Only `train.csv` is used (included at `data/train.csv`). It is split 80/20 (stratified);
the 20 % test portion (`data/unseen_data.csv`, 179 rows) provides the **unseen** records for Tasks 2-4.

## Repository Structure
| File | Purpose |
|---|---|
| `task1_train_model.py` | Task 1 - preprocess, train, evaluate accuracy, save `model.pkl` |
| `task2_online_inference.py` | Task 2 - predict 1 unseen passenger |
| `task3_batch_inference.py` | Task 3 - predict 20 unseen passengers -> `batch_predictions.csv` |
| `task4_streaming_inference.py` | Task 4 - 10 records, one per second |
| `app.py` | Task 5 - FastAPI app (`GET /health`, `POST /predict`, Pydantic validation) |
| `client.py` | Task 5 - Python client testing the API |
| `model.pkl` | Saved trained pipeline |
| `batch_predictions.csv` | Batch inference output |
| `requirements.txt` | Dependencies |
| `screenshots/` | Output screenshots |
| `Assignment_Report.pdf` | Report with observations (Task 6) |

## Execution Steps
```bash
# 1. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the tasks in order
python task1_train_model.py          # trains and creates model.pkl + data/unseen_data.csv
python task2_online_inference.py
python task3_batch_inference.py      # creates batch_predictions.csv
python task4_streaming_inference.py

# 4. serve the model
uvicorn app:app --reload             # Swagger UI: http://127.0.0.1:8000/docs

# 5. in a second terminal, test the API
python client.py
```

## Results
- **Test accuracy:** 79.89 % (RandomForest, 179 held-out records)
- **Online inference:** 1 unseen passenger predicted with probability shown (see `screenshots/2_online_inference.png`)
- **Batch inference:** 20 predictions saved in `batch_predictions.csv`
- **Streaming inference:** 10 records, one prediction per second
- **FastAPI:** `/health` -> `{"status":"ok","model_loaded":true}`;
  valid `/predict` -> HTTP 200 with prediction; invalid `/predict` -> HTTP 422 with validation errors.

Example request body for `POST /predict`:
```json
{"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0, "Parch": 0, "Fare": 100.0, "Embarked": "S"}
```

## Observations
See `Assignment_Report.pdf` for answers to the Task 6 questions.
