# Aerosight — Flight Delay Intelligence

A flight delay prediction and prescriptive analytics web app powered by LightGBM, SHAP, and 7M US domestic flight records.

## Live Demo
[aerosight-webapp.vercel.app](https://aerosight-webapp.vercel.app)

## Stack
- **Model:** LightGBM + Optuna + SHAP
- **Data:** 7M US domestic flights (2024 BTS)
- **Weather:** Open-Meteo API
- **Backend:** FastAPI on HuggingFace Spaces
- **Frontend:** Vanilla HTML/CSS/JS on Vercel

## Features
- Real-time delay probability prediction
- SHAP explainability — know why a flight is at risk
- Prescriptive action plan with cost estimates
- US Airport delay risk map
- Network operations dashboard
- Model performance breakdown
