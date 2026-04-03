# ⚡ StudyPulse – Smart Study Time Predictor

> A machine-learning powered study optimization platform that identifies your peak productivity hours, generates personalized daily schedules, and monitors burnout risk.

**Hackathon:** StudyPulse – Smart Study Time Predictor | Tekworks Innovation Arena  
**Stack:** Python · Pandas · Scikit-Learn · Streamlit · Matplotlib · Seaborn

---

## 🎯 Problem Solved

Traditional study timetables assume students have uniform focus all day. StudyPulse uses behavioral data and ML to:

- 🏆 Identify your **golden study hours** (peak productivity windows)
- 📅 Generate a **personalized daily schedule** with smart subject ordering
- 🎓 Provide an **adaptive exam preparation engine**
- 🔥 Detect **burnout risk** before it happens
- 🎮 Track productivity with a **gamification score system**

---

## 🏗️ Project Structure

```
studypulse/
├── app.py                          # Main Streamlit dashboard
├── data/
│   ├── generate_dataset.py         # Synthetic 90-day study log generator
│   └── student_study_log.csv       # Auto-generated on first run
├── models/
│   ├── train_models.py             # ML pipeline (3 models)
│   └── studypulse_models.pkl       # Auto-saved trained models
├── utils/
│   ├── recommender.py              # Schedule & recommendation engine
│   └── visualizations.py          # All chart/plot functions
├── requirements.txt
└── README.md
```

---

## 🤖 ML Models

| Model | Algorithm | Target | Performance |
|-------|-----------|--------|-------------|
| Focus Predictor | Random Forest Regressor | Focus Level (0-10) | R² ≈ 0.85 |
| Peak Hour Classifier | Gradient Boosting | Is this a peak hour? | Accuracy ≈ 0.82 |
| Burnout Risk Scorer | Random Forest Regressor | Burnout Risk (0-1) | R² ≈ 0.88 |

### Key Features Used
- Cyclical time encoding (hour sin/cos)
- 7-day rolling averages (focus, sleep, stress)
- Subject difficulty × exam pressure interactions
- Sleep × focus interaction terms
- Day-of-week & time-of-day encoding

---

## 📊 Dataset

**Synthetic dataset** generated with realistic patterns including:
- 90-day study log (≈245 sessions)
- Chronotype simulation (morning / neutral / evening learner)
- Productivity curve based on circadian rhythm research
- Exam pressure dynamics (urgency increases as exam approaches)
- Cumulative stress & fatigue modeling
- 18 features: date, hour, subject, focus level, sleep, stress, burnout risk, etc.

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the App
```bash
streamlit run app.py
```

The app will:
1. Auto-generate the synthetic dataset on first run
2. Train and cache ML models automatically
3. Open at `http://localhost:8501`

---

## 🖥️ Dashboard Features

### Tab 1 – Today's Schedule
- Visual timeline of your day (Gantt-style)
- Hourly productivity forecast bar chart
- Session cards with method, Pomodoro style, and exam notes

### Tab 2 – Analytics
- 30-day focus/sleep/stress trend
- Subject performance radar chart
- Sleep vs focus scatter (coloured by stress)
- Time-of-day focus distribution

### Tab 3 – AI Predictions
- Live model metrics (R², MAE, accuracy)
- Feature importance chart
- Interactive focus score predictor

### Tab 4 – Exam Prep
- Adaptive strategy per subject (4 phases)
- Daily hour recommendations
- Focus area and study method suggestions

### Tab 5 – Burnout Radar
- Risk score with colour-coded alert
- 90-day burnout timeline
- Recovery action checklist
- Gamification badge & score

---

## 🧠 Recommendation Logic

### Schedule Generation
1. Weight subjects by `difficulty × exam_urgency`
2. Assign hardest/most urgent subjects to peak hours
3. Select Pomodoro variant based on burnout risk
4. Generate blocks with subject-specific study methods

### Pomodoro Variants
| Mode | Work | Break |
|------|------|-------|
| High Focus | 50 min | 10 min |
| Medium Focus | 35 min | 10 min |
| Low Focus | 25 min | 10 min |
| Burnout Risk | 20 min | 10 min |

---

## 📦 Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 📝 Assumptions & Notes

- All data is **synthetic** — no real student data used
- Dataset generated with `SEED=42` for reproducibility
- Models are cached after first training run
- Exam dates in the sidebar default to 2025-08-15; adjust per subject
- The productivity curve follows circadian rhythm research patterns

---

## 🏆 Evaluation Alignment

| Criterion | Implementation |
|-----------|---------------|
| Relevance | Directly targets focus, burnout, exam scheduling |
| Practical usefulness | Actionable daily schedule + method recommendations |
| Innovation | Chronotype modeling, circadian productivity curve, gamification |
| Model logic | 3 ML models with feature engineering & cross-validation |
| Technical quality | Modular code, cached models, clean pipeline |
| UX & clarity | Dark theme dashboard, tabbed layout, metric cards |

---

*Built for the Tekworks StudyPulse Hackathon — April 2026*
