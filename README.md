# churn-api-deployment

# Task 3 — Model Deployment: FastAPI + Docker

**Alfido Tech Internship**

Wraps the trained churn prediction model (from Task 1) in a FastAPI REST API and containerizes it with Docker.

---

## Quick Demo

**Health check:**
```bash
curl http://localhost:8000/health
# {"status":"ok","model":"RandomForestClassifier","features":19}
```

**Predict churn:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

**Expected response:**
```json
{
  "churn": true,
  "probability": 0.7312,
  "confidence": "high",
  "risk_level": "HIGH RISK",
  "recommendation": "Immediate retention action recommended — offer a contract upgrade or discount."
}
```

---

## Project Structure

```
task3-deployment/
├── app/
│   ├── __init__.py
│   └── main.py              
├── notebook/
│   └── deployment_notebook.ipynb
├── model.pkl                
├── scaler.pkl               
├── feature_names.pkl       
├── Dockerfile
├── requirements.txt
├── sample_request.json
└── README.md
```

---

## Setup & Run (Without Docker)

### 1. Prerequisites
Complete Task 1 first — it generates `model.pkl`, `scaler.pkl`, `feature_names.pkl`.
Copy those three files into this folder.

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate     
pip install -r requirements.txt
```

### 3. Start the server
```bash
uvicorn app.main:app --reload
```

### 4. Open interactive docs
Visit: http://localhost:8000/docs

---

## Setup & Run (With Docker)

### 1. Build the image
```bash
docker build -t churn-api .
```

### 2. Run the container
```bash
docker run -d -p 8000:8000 --name churn-api churn-api
```

### 3. Test it
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

### 4. Stop the container
```bash
docker stop churn-api && docker rm churn-api
```

---

## API Endpoints

| Method | Endpoint   | Description                        |
|--------|------------|------------------------------------|
| GET    | `/`        | API info and available endpoints   |
| GET    | `/health`  | Model status and feature count     |
| POST   | `/predict` | Predict churn for a customer       |
| GET    | `/docs`    | Swagger UI (interactive testing)   |

---

## Input Fields

| Field            | Type  | Description                                      |
|------------------|-------|--------------------------------------------------|
| gender           | int   | 0 = Female, 1 = Male                            |
| SeniorCitizen    | int   | 0 = No, 1 = Yes                                 |
| tenure           | float | Months with company                             |
| Contract         | int   | 0 = Month-to-month, 1 = One year, 2 = Two year |
| MonthlyCharges   | float | Current monthly bill (USD)                      |
| TotalCharges     | float | Total amount billed (USD)                       |
| InternetService  | int   | 0 = DSL, 1 = Fiber optic, 2 = None             |
| ... (19 total)   |       | See /docs for full schema                       |

---

## Environment

- Python 3.11
- FastAPI 0.111.0
- scikit-learn 1.5.0
- Docker (any recent version)

---

*Alfido Tech Internship — Task 3 of 3*
