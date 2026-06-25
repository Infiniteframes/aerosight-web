Flight Delay Risk Predictor:
A machine learning system that predicts the delay risk of a flight before it departs, using historical flight data. The model outputs a delay-risk probability (e.g. "60.6% High Risk") along with the key factors driving that prediction, helping flag flights that need proactive attention.

📊 Model Performance
MetricScoreNotesAccuracy72.2%Real-world distributionAUC Score0.7247ROC-AUCF1 Score0.4498Balanced F1Precision38.0%1.9× better than randomRecall55.1%Delays caught

🤖 Models Compared
LightGBM (Optuna-tuned) — best performer (AUC ≈ 0.72), used in final pipeline
Random Forest
Gradient Boosting
Logistic Regression (baseline)

🔍 Key Predictive Features (SHAP-based)
Ranked by impact on delay risk:

Airline Reliability (~38%)
Weather Condition (~19%)
Historical Route Delays (~18%)
Route Distance (~13%)
Season / Month (~11%)
Departure Hour, Origin Airport, Temperature, Precipitation

🖥️ Features
Delay risk dashboard — per-flight risk score, confidence level (based on similar historical routes), and route preview
Explainable predictions — SHAP values show why a flight is flagged as high-risk
Operational recommendations — auto-generated alerts (e.g. notify passengers, pre-position gate agents, flag cascade-delay risk on connections)

📁 Dataset
Trained on a public flight delay dataset from Kaggle.

Built as a tool to help airlines/ops teams flag high-risk flights proactively rather than reactively.
