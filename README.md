# TerraAlert 

> Predicting wildfires, floods, earthquakes, and tornadoes using real government data and machine learning with live deployment, SHAP explainability, and automated emergency response.

**Live Platform:** [terraalert-app.web.app](https://terraalert-app.web.app)  
**API Docs:** [localhost:8000/docs](http://localhost:8000/docs) (run locally)  
**Author:** Indrayani Vijaysinh Bhosale   

---

## What is TerraAlert?

TerraAlert is a full-stack disaster intelligence platform that compares machine learning architectures across four disaster types to determine which performs best per hazard and whether a unified model can outperform specialized ones.

It combines:
- **Real government data** from USGS, NASA FIRMS, and NOAA
- **7 trained ML models** across 4 disaster types
- **Live predictions** via FastAPI backend
- **SHAP explainability** for every prediction
- **Automated email alerts** for critical events
- **Resource routing** using Haversine distance
- **React dashboard** with live map and Command Center mode

---

## Project Structure

```
terraalert/
├── backend/
│   ├── main.py              # FastAPI v6.0 — all 4 disaster endpoints
│   └── .env                 # API keys (not committed)
├── terraalert-frontend/
│   └── src/
│       └── App.js           # React dashboard
├── notebooks/
│   ├── 01_earthquake_eda.ipynb
│   ├── 02_wildfire_eda.ipynb
│   ├── 03_flood_eda.ipynb
│   ├── 04_wildfire_models.ipynb
│   ├── 05_flood_models.ipynb
│   ├── 06_earthquake_models.ipynb
│   ├── 07_unified_model.ipynb
│   ├── 08_tornado_eda.ipynb
│   └── 09_tornado_models.ipynb
├── models/                  # Trained model files (not committed — see below)
└── data/                    # Datasets (not committed — see below)
```

---

## Datasets

| Disaster | Source | Records |
|---|---|---|
| Earthquake | USGS Catalog 2020-2024 | 8,678 |
| Earthquake | USGS Historical 1990-2019 | 1,434 |
| Wildfire | 1.88M US Wildfires (Kaggle) | 1,880,465 |
| Wildfire | NASA FIRMS Hotspots | 1,145 |
| Flood | NOAA Storm Events 2020-2024 | 40,796 |
| Tornado | NOAA Tornado Records 1950-2023 | 70,022 |

**Total: Over 2 million records across 6 datasets**

Datasets are not committed to this repo due to size. Download instructions are in each notebook.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Models | scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch |
| Explainability | SHAP TreeExplainer |
| Backend | FastAPI, Uvicorn, joblib |
| Frontend | React.js, Leaflet.js |
| Live Data | USGS API, NASA FIRMS API, NOAA API |
| Weather | OpenWeatherMap API |
| Email Alerts | Gmail SMTP |
| Deployment | Firebase Hosting |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict/wildfire` | POST | RF prediction with live weather boost |
| `/predict/flood` | POST | RF severity classification |
| `/predict/earthquake` | POST | TFT magnitude forecasting |
| `/predict/tornado` | POST | RF severity classification |
| `/explain/wildfire` | POST | SHAP feature importance per prediction |
| `/live/earthquakes` | GET | Live USGS feed last 48 hours |
| `/weather/{lat}/{lon}` | GET | Current weather conditions |
| `/resources` | GET | Emergency unit inventory |
| `/report/incident` | POST | Submit field incident report |
| `/history/predictions` | GET | Last 50 prediction log entries |

---

## Running Locally

### Prerequisites

```bash
Python 3.13+
Node.js 23+
```

### 1. Clone the repo

```bash
git clone https://github.com/IndrayaniBhosale/terraalert.git
cd terraalert
```

### 2. Set up environment variables

Create `backend/.env`:

```
WEATHER_API_KEY=your_openweathermap_key
EMAIL_SENDER=your_gmail@gmail.com
EMAIL_PASSWORD=your_gmail_app_password
EMAIL_RECIPIENT=your_email@gmail.com
```

### 3. Install Python dependencies

```bash
pip install fastapi uvicorn joblib torch scikit-learn xgboost lightgbm catboost shap pandas numpy python-dotenv requests
```

### 4. Start the backend

```bash
cd backend
python3 -m uvicorn main:app --reload --port 8000
```

Backend runs at `http://localhost:8000`  
API docs at `http://localhost:8000/docs`

### 5. Install frontend dependencies

```bash
cd terraalert-frontend
npm install
```

### 6. Start the frontend

```bash
npm start
```

Dashboard runs at `http://localhost:3000`

---

## Running the Notebooks

```bash
cd notebooks
python3 -m jupyter notebook
```

Run notebooks in order:
1. `01_earthquake_eda.ipynb` — Download and explore USGS data
2. `02_wildfire_eda.ipynb` — Explore 1.88M wildfire records
3. `03_flood_eda.ipynb` — Explore NOAA flood data
4. `04_wildfire_models.ipynb` — Train and evaluate wildfire models
5. `05_flood_models.ipynb` — Train and evaluate flood models (data leakage fixed)
6. `06_earthquake_models.ipynb` — LSTM vs TFT vs PETformer comparison
7. `07_unified_model.ipynb` — Multi-hazard unified model experiment
8. `08_tornado_eda.ipynb` — Explore NOAA tornado records 1950-2023
9. `09_tornado_models.ipynb` — Train and evaluate tornado models

---

## Model Files

Trained model files are not committed due to size. After running the notebooks they will be saved to:

```
models/
├── rf_wildfire.joblib
├── rf_flood.joblib
├── rf_tornado.joblib
├── tft_earthquake.pt
└── earthquake_scaler.joblib
```

---

## Dashboard Features

- **Live map** — earthquakes, wildfires, floods, tornadoes color-coded by disaster type
- **Disaster isolation** — filter to show only one disaster type at a time
- **SHAP popups** — click any wildfire marker to see why it was flagged
- **Deploy Resources** — click CRITICAL events to route nearest emergency units
- **Command Center** — operational view with threat matrix and prediction log
- **Field reports** — click map to submit incident reports
- **Email alerts** — automatic CRITICAL alerts with resource allocation table

---

## Disaster Colors

| Color | Disaster |
|---|---|
| 🔵 Blue | Earthquake |
| 🟠 Orange | Wildfire |
| 🟢 Green | Flood |
| 🟣 Purple | Tornado |

---

## Acknowledgments

- USGS Earthquake Hazards Program
- NASA FIRMS (Fire Information for Resource Management System)
- NOAA National Centers for Environmental Information
- Kaggle — 1.88M US Wildfires dataset
- OpenWeatherMap API

