# ML Model Deployment & Model Serving Using FastAPI

A machine learning project demonstrating model training, multiple inference strategies, and REST API deployment using **Python, Scikit-learn, and FastAPI**.

## 1. Project Overview

This project focuses on training a classification model using the Titanic dataset and implementing three inference strategies: **online inference, batch inference, and streaming inference**. It also demonstrates REST API model serving using FastAPI.

The model is trained **once**, and the saved pipeline (`model.pkl`), containing preprocessing and the classifier, is reused across all inference tasks and API requests.

## 2. Dataset Information

**Dataset:** Titanic – Machine Learning from Disaster (Kaggle)

**Dataset Link:** https://www.kaggle.com/competitions/titanic/data

- **Input file:** `data/train.csv`
- **Data usage:** Only `train.csv` is used.
- **Train-test split:** 80/20 (stratified).
- **Unseen data:** The 20% test portion is saved as `data/unseen_data.csv`.
- **Unseen records:** 179 rows.

The unseen dataset is used to perform inference for **Tasks 2–4**, ensuring predictions are made on records not used during model training.

## 3. Repository Structure

The repository contains the following files and directories:

| File / Directory | Description |
|---|---|
| `task1_train_model.py` | Task 1 – Preprocess data, train the model, evaluate accuracy, and save `model.pkl`. |
| `task2_online_inference.py` | Task 2 – Predict the outcome for 1 unseen passenger. |
| `task3_batch_inference.py` | Task 3 – Generate predictions for 20 unseen passengers and save them to `batch_predictions.csv`. |
| `task4_streaming_inference.py` | Task 4 – Process 10 records, one per second. |
| `app.py` | Task 5 – FastAPI application with `GET /health`, `POST /predict`, and Pydantic validation. |
| `client.py` | Task 5 – Python client for testing the API. |
| `model.pkl` | Saved machine learning pipeline. |
| `batch_predictions.csv` | Output file containing batch inference predictions. |
| `requirements.txt` | Required project dependencies. |
| `screenshots/` | Screenshots showing the execution outputs. |
| `Assignment_Report.pdf` | Report containing observations and answers for Task 6. |

## 4. Installation and Execution Guide

Follow the steps below to set up the environment and execute the project.

### Step 1: Create a Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Execute the Inference Tasks

Run the scripts in the specified order.

```bash
python task1_train_model.py          # trains and creates model.pkl + data/unseen_data.csv
python task2_online_inference.py
python task3_batch_inference.py      # creates batch_predictions.csv
python task4_streaming_inference.py
```

### Step 4: Start the FastAPI Server

```bash
uvicorn app:app --reload
```

**Swagger UI:** http://127.0.0.1:8000/docs

### Step 5: Test the API

Open a second terminal and execute:

```bash
python client.py
```

## 5. Results and Performance

The project successfully demonstrates model training, inference, and API deployment.

| Evaluation Metric | Result |
|---|---|
| Test Accuracy | **79.89 %** |
| Classification Algorithm | RandomForest |
| Held-out Test Records | 179 |
| Online Inference | 1 unseen passenger predicted with probability shown |
| Batch Inference | 20 predictions saved in `batch_predictions.csv` |
| Streaming Inference | 10 records processed, one prediction per second |

**Screenshot reference:** `screenshots/2_online_inference.png` contains the online inference output.

### FastAPI Validation Results

| Endpoint / Request | Expected Result |
|---|---|
| `/health` | `{"status":"ok","model_loaded":true}` |
| Valid `/predict` request | HTTP 200 with prediction |
| Invalid `/predict` request | HTTP 422 with validation errors |

## 6. Sample API Request

The following JSON payload demonstrates a sample request to the `POST /predict` endpoint.

```json
{"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0, "Parch": 0, "Fare": 100.0, "Embarked": "S"}
```

## 7. Observations and Analysis

The project illustrates how a trained machine learning pipeline can be reused across different inference approaches and integrated into a REST API.

- **Online inference:** Produces a prediction for an individual unseen passenger.
- **Batch inference:** Generates predictions for multiple unseen records and stores the results in a CSV file.
- **Streaming inference:** Processes records sequentially, with one prediction generated per second.
- **API deployment:** Provides health monitoring, prediction functionality, and input validation through FastAPI.

For detailed observations and answers to the Task 6 questions, refer to **`Assignment_Report.pdf`**.

---

*Project completed using Python, machine learning, and FastAPI to demonstrate practical model deployment and serving strategies.*
