"""
StudyPulse – Smart Study Time Predictor
Streamlit Dashboard
Run: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, date

from data.generate_dataset import generate_dataset
from models.train_models import train_models, engineer_features
from utils.recommender import (
    hourly_productivity_scores, generate_daily_schedule,
    exam_prep_strategy, burnout_analysis, gamification_score,
)
from utils.visualizations import (
    plot_hourly_productivity, plot_weekly_trend, plot_subject_radar,
    plot_tod_distribution, plot_sleep_focus, plot_burnout_timeline,
    plot_daily_schedule, plot_feature_importance,
)

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="StudyPulse",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* 🌌 MAIN BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #E6EDF3;
    font-family: 'Space Grotesk', sans-serif;
}

/* 🌌 SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* 📝 SIDEBAR TEXT & LABELS */
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
    color: #FFFFFF !important;
}

/* 🔘 INPUT BOXES (Sidebar & Main) */
[data-testid="stSidebar"] input, 
[data-testid="stSidebar"] textarea,
.stNumberInput input, 
.stTextInput input {
    color: #000000 !important; /* Black text for readability inside white boxes */
    background-color: #ffffff !important;
    border-radius: 8px;
}

/* 🎯 SELECTBOX & DROPDOWN FIX (CRITICAL) */

/* 1. The container for the selected value */
div[data-baseweb="select"] > div {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.2);
}

/* 2. The actual text of the selected item */
div[data-baseweb="select"] span {
    color: #ffffff !important;
}

/* 3. The dropdown menu popup (The Listbox) */
ul[role="listbox"] {
    background-color: #1e293b !important;
    border: 1px solid #00DBDE !important;
}

/* 4. Individual options inside the dropdown */
li[role="option"] {
    color: #ffffff !important;
    background-color: #1e293b !important;
}

/* 5. Hover state for options */
li[role="option"]:hover {
    background-color: #00DBDE !important;
    color: #0f172a !important; /* Dark text on neon highlight */
}

/* 6. Multi-select tags (Pills) */
span[data-baseweb="tag"] {
    background-color: #ff416c !important; 
    color: white !important;
}

/* 7. Dropdown icons (Arrows) */
svg[data-baseweb="icon"] {
    fill: #ffffff !important;
}

/* ⚡ TITLE & HEADERS */
.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00DBDE, #FC00FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.section-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: #ffffff;
    text-shadow: 0 0 8px rgba(0,219,222,0.6);
    border-bottom: 1px solid rgba(255,255,255,0.2);
    margin: 20px 0 10px;
}

/* 📊 METRIC CARDS */
.metric-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 18px;
    border: 1px solid rgba(255,255,255,0.1);
    transition: 0.3s;
}

.metric-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

.metric-val {
    font-size: 2rem;
    font-weight: bold;
    background: linear-gradient(90deg, #00DBDE, #FC00FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'JetBrains Mono';
}

.metric-label {
    font-size: 0.8rem;
    color: #9ca3af;
}

/* 📅 SCHEDULE BLOCKS */
.schedule-block {
    background: rgba(0,0,0,0.5);
    border-radius: 10px;
    padding: 12px;
    margin: 6px 0;
    border-left: 5px solid #00DBDE;
    transition: 0.3s;
}

.burnout-low      { border-left-color: #22c55e !important; }
.burnout-medium   { border-left-color: #facc15 !important; }
.burnout-high     { border-left-color: #fb923c !important; }
.burnout-critical { border-left-color: #ef4444 !important; }

/* 🏆 BADGE CARD */
.badge-card {
    background: linear-gradient(135deg, #1e293b, #334155);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
}

/* 📑 TABS */
.stTabs [data-baseweb="tab"] {
    font-size: 15px;
    color: #aaa;
}

.stTabs [aria-selected="true"] {
    color: #00DBDE !important;
    border-bottom: 3px solid #00DBDE;
}

/* 🔘 PRIMARY BUTTON */
button[kind="primary"] {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    border-radius: 10px;
    border: none;
    color: white !important;
}

button[kind="primary"]:hover {
    transform: scale(1.05);
}

/* 🎚️ SLIDERS */
.stSlider label {
    color: #E6EDF3 !important;
}
/* 🚨 THE INVISIBLE TEXT OVERRIDE 🚨 */

/* 1. Force the text color inside the selectbox (closed state) */
div[data-baseweb="select"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important; /* Forces color on some browsers */
}

/* 2. Target the search input text specifically */
input[aria-autocomplete="list"] {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* 3. Target the dropdown menu (open state) */
[data-baseweb="popover"] div, 
[data-baseweb="menu-item"], 
[role="option"] {
    color: #ffffff !important;
    background-color: #1e293b !important;
}

/* 4. Ensure hover doesn't make it disappear again */
[data-baseweb="menu-item"]:hover, 
[role="option"]:hover {
    background-color: #334155 !important;
    color: #00DBDE !important;
}

/* 5. Fix for the 'X' (clear) and 'Arrow' icons */
[data-baseweb="select"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}

/* 6. Fix for placeholders if they are still dark */
div[data-baseweb="select"] input::placeholder {
    color: #9ca3af !important;
    -webkit-text-fill-color: #9ca3af !important;
}
/* 🎓 EXAM PREP SUBJECT NAME COLOR FIX */

/* Targets the text inside the expander header */
.stExpander [data-testid="stExpanderHeader"] p {
    color: #00DBDE !important; /* Neon Cyan */
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Optional: Change color when the expander is hovered */
.stExpander [data-testid="stExpanderHeader"]:hover p {
    color: #FC00FF !important; /* Neon Pink/Purple on hover */
}

/* Ensure the arrow icon matches the new text color */
.stExpander [data-testid="stExpanderHeader"] svg {
    fill: #00DBDE !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Initialisation ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_or_generate_data():
    csv = "data/student_study_log.csv"
    if os.path.exists(csv):
        return pd.read_csv(csv)
    df = generate_dataset()
    df.to_csv(csv, index=False)
    return df


@st.cache_resource(show_spinner=False)
def load_or_train_models(df):
    pkl = "models/studypulse_models.pkl"
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            return pickle.load(f)
    arts = train_models(df)
    os.makedirs("models", exist_ok=True)
    with open(pkl, "wb") as f:
        pickle.dump(arts, f)
    return arts


# ── Sidebar – Student Profile ─────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ StudyPulse")
    st.markdown("*Smart Study Time Predictor*")
    st.divider()

    st.markdown("### 👤 Your Profile")
    student_name = st.text_input("Name", value="Alex")
    sleep_tonight = st.slider("Sleep last night (hrs)", 3.0, 10.0, 7.0, 0.5)
    stress_today  = st.slider("Today's stress level (1-10)", 1, 10, 4)
    hours_to_study = st.slider("Hours available today", 1.0, 8.0, 4.0, 0.5)

    st.markdown("### 📚 Subjects Today")
    SUBJECTS = ["Mathematics", "Physics", "Chemistry", "Biology", "History", "English"]
    selected_subjects = st.multiselect("Choose subjects", SUBJECTS, default=["Mathematics", "Physics", "English"])

    st.markdown("### 📅 Exam Dates")
    exam_dates = {}
    for subj in selected_subjects:
        d = st.date_input(f"{subj}", value=date(2026, 5, 10), key=f"exam_{subj}")
        days_left = (d - date.today()).days
        exam_dates[subj] = max(days_left, 0)

    st.divider()
    st.markdown("### 📊 Previous Scores")
    prev_scores = {}
    for subj in selected_subjects:
        prev_scores[subj] = st.slider(subj, 0, 100, 65, 5, key=f"score_{subj}")


# ── Main Content ──────────────────────────────────────────────────────────────

# Header
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown(f'<div class="main-title">⚡ StudyPulse</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">Smart Study Time Predictor &nbsp;|&nbsp; Hello, {student_name}! &nbsp;|&nbsp; {datetime.now().strftime("%A, %d %B %Y")}</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Load Data & Models ────────────────────────────────────────────────────────
with st.spinner("🔄 Loading data & AI models..."):
    df = load_or_generate_data()
    artifacts = load_or_train_models(df)

results = artifacts["results"]

# ── KPI Cards ─────────────────────────────────────────────────────────────────
hourly_scores = hourly_productivity_scores(sleep_tonight, stress_today)
top_hour = max(hourly_scores, key=hourly_scores.get)
top_score = hourly_scores[top_hour]

avg_focus_7d = df.tail(21)["focus_level"].mean()
burnout_score = df.tail(14)["burnout_risk"].mean()
avg_sleep_7d  = df.tail(21)["sleep_hours"].mean()
streak_days   = min(30, int((1 - burnout_score) * 30))
sessions_done = len(df.tail(30))

gam = gamification_score(streak_days, avg_focus_7d, sessions_done, burnout_score)

k1, k2, k3, k4, k5 = st.columns(5)
metrics = [
    (k1, f"{top_hour:02d}:00", "🏆 Golden Study Hour"),
    (k2, f"{top_score:.1f}/10", "⚡ Peak Focus Score"),
    (k3, f"{avg_focus_7d:.1f}/10", "📊 7-Day Avg Focus"),
    (k4, f"{burnout_score*100:.0f}%", "🔥 Burnout Risk"),
    (k5, f"{gam['score']}", f"🎮 {gam['badge']} Rank: {gam['rank']}"),
]
for col, val, label in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 Today's Schedule",
    "📈 Analytics",
    "🧠 AI Predictions",
    "🎓 Exam Prep",
    "🔥 Burnout Radar",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – TODAY'S SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">⚡ Your Personalised Study Schedule</div>', unsafe_allow_html=True)

    if not selected_subjects:
        st.warning("Please select at least one subject in the sidebar.")
    else:
        schedule = generate_daily_schedule(
            subjects=selected_subjects,
            hours_available=hours_to_study,
            focus_scores_by_hour=hourly_scores,
            exam_dates=exam_dates,
            sleep_hours=sleep_tonight,
            stress_level=stress_today,
            burnout_risk=burnout_score,
        )

        # Schedule timeline chart
        fig_sched = plot_daily_schedule(schedule)
        if fig_sched:
            st.pyplot(fig_sched, use_container_width=True)

        # Hourly productivity
        fig_prod = plot_hourly_productivity(hourly_scores)
        st.pyplot(fig_prod, use_container_width=True)

        st.markdown('<div class="section-header">📋 Session Breakdown</div>', unsafe_allow_html=True)
        cols = st.columns(min(len(schedule), 3))
        for i, block in enumerate(schedule):
            with cols[i % len(cols)]:
                risk_class = (
                    "burnout-high" if burnout_score > 0.6
                    else "burnout-medium" if burnout_score > 0.4
                    else "burnout-low"
                )
                st.markdown(f"""
                <div class="schedule-block {risk_class}">
                    <b>{block['hour']:02d}:00</b> – <b>{block['subject']}</b><br>
                    ⏱ {block['duration_min']} min + {block['break_after_min']} min break<br>
                    ⚡ Focus score: <b>{block['focus_score']:.1f}</b><br>
                    📝 Method: <i>{block['method']}</i><br>
                    🍅 Mode: {block['pomodoro_style']}<br>
                    {block['notes']}
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-header">📈 Weekly Trend</div>', unsafe_allow_html=True)
        fig_trend = plot_weekly_trend(df)
        st.pyplot(fig_trend, use_container_width=True)

        st.markdown('<div class="section-header">😴 Sleep vs Focus</div>', unsafe_allow_html=True)
        fig_sf = plot_sleep_focus(df)
        st.pyplot(fig_sf, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">🎯 Subject Radar</div>', unsafe_allow_html=True)
        fig_radar = plot_subject_radar(df)
        st.pyplot(fig_radar, use_container_width=True)

        st.markdown('<div class="section-header">🌅 Time-of-Day Focus</div>', unsafe_allow_html=True)
        fig_tod = plot_tod_distribution(df)
        st.pyplot(fig_tod, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – AI PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">🧠 Model Metrics</div>', unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    model_metrics = [
        (mc1, f"{results['focus_r2']:.3f}", "Focus R²"),
        (mc2, f"{results['focus_mae']:.3f}", "Focus MAE"),
        (mc3, f"{results['peak_accuracy']:.3f}", "Peak Hour Accuracy"),
        (mc4, f"{results['burnout_r2']:.3f}", "Burnout R²"),
    ]
    for col, val, label in model_metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="font-size:1.5rem">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    feat_imp = results.get("focus_feature_importance")
    if feat_imp is not None:
        with st.expander("📊 Feature Importance"):
            fig_fi = plot_feature_importance(feat_imp)
            st.pyplot(fig_fi, use_container_width=True)

    st.markdown('<div class="section-header">🔮 Predict Focus for a Session</div>', unsafe_allow_html=True)
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        pred_hour    = st.selectbox("Session Hour", list(range(6, 24)), index=4)
        pred_subject = st.selectbox("Subject", SUBJECTS)
    with pcol2:
        pred_sleep  = st.slider("Sleep (hrs)", 3.0, 10.0, sleep_tonight, 0.5, key="pred_sleep")
        pred_stress = st.slider("Stress (1-10)", 1, 10, stress_today, key="pred_stress")
    with pcol3:
        pred_exam_days = st.number_input("Days to exam", 0, 365, 30)
        pred_energy = st.slider("Energy level (0-1)", 0.0, 1.0, 0.7, 0.05)

    if st.button("🔮 Predict My Focus Score", type="primary"):
        from models.train_models import engineer_features as ef
        # Build a single-row input
        tod_map = {h: ("Morning" if 5<=h<12 else "Afternoon" if 12<=h<17 else "Evening" if 17<=h<21 else "Night") for h in range(24)}
        dow_map = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}
        input_row = pd.DataFrame([{
            "date": datetime.now().strftime("%Y-%m-%d"),
            "day_of_week": dow_map[datetime.now().weekday()],
            "hour": pred_hour,
            "time_of_day": tod_map[pred_hour],
            "subject": pred_subject,
            "subject_difficulty": {"Mathematics":9,"Physics":8,"Chemistry":7,"Biology":6,"History":4,"English":3}[pred_subject],
            "study_hours": 1.5,
            "focus_level": 5.0,
            "sleep_hours": pred_sleep,
            "break_duration_min": 10.0,
            "days_to_exam": pred_exam_days,
            "exam_pressure": float(np.clip(1 - pred_exam_days/30, 0, 1)),
            "previous_score": 70.0,
            "stress_level": pred_stress,
            "burnout_risk": 0.3,
            "is_weekend": int(datetime.now().weekday() >= 5),
            "cumulative_stress": 0.3,
            "energy_level": pred_energy,
        }])

        combined = pd.concat([df, input_row], ignore_index=True)
        combined_feat, _, _, _ = ef(combined)
        feat_row = combined_feat.iloc[[-1]][artifacts["focus_features"]]
        pred_focus = float(artifacts["focus_model"].predict(feat_row)[0])
        pred_peak  = int(artifacts["peak_model"].predict(combined_feat.iloc[[-1]][artifacts["peak_features"]])[0])

        colour = "#3FB950" if pred_focus >= 7 else "#D29922" if pred_focus >= 5 else "#F78166"
        st.markdown(f"""
        <div class="metric-card" style="border-color:{colour}; margin-top:12px;">
            <div class="metric-val" style="color:{colour}; font-size:2.5rem">{pred_focus:.1f}/10</div>
            <div class="metric-label">Predicted Focus Score</div>
            <div style="color:{colour}; margin-top:8px; font-size:0.9rem">
                {'🌟 PEAK HOUR — perfect time to study!' if pred_peak else '⚠️ Not a peak hour — consider rescheduling.'}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – EXAM PREP
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">🎓 Adaptive Exam Preparation Engine</div>', unsafe_allow_html=True)

    for subj in selected_subjects:
        plan = exam_prep_strategy(
            subject=subj,
            days_until_exam=exam_dates.get(subj, 30),
            previous_score=prev_scores.get(subj, 65),
            avg_focus=avg_focus_7d,
        )
        urgency_color = (
            "#F78166" if plan["days_until_exam"] <= 7
            else "#D29922" if plan["days_until_exam"] <= 21
            else "#3FB950"
        )
        with st.expander(f"{subj} — {plan['phase']} ({plan['days_until_exam']} days left)", expanded=plan["days_until_exam"] <= 14):
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="font-size:1.4rem; color:{urgency_color}">{plan['days_until_exam']}d</div>
                    <div class="metric-label">Days to Exam</div>
                </div>""", unsafe_allow_html=True)
            with ec2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="font-size:1.4rem">{plan['recommended_daily_hours']}h</div>
                    <div class="metric-label">Daily Study Hours</div>
                </div>""", unsafe_allow_html=True)
            with ec3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="font-size:1.4rem">{plan['previous_score']:.0f}%</div>
                    <div class="metric-label">Previous Score — {plan['performance_tier']}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"**📝 Focus Areas:** {' · '.join(plan['focus_areas'])}")
            st.markdown(f"**🔄 Revision vs New:** {plan['revision_percentage']}% revision / {plan['new_content_percentage']}% new")
            st.markdown(f"**🧠 Primary Method:** {plan['primary_method']}")
            st.progress(plan["revision_percentage"] / 100)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 – BURNOUT RADAR
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">🔥 Burnout Risk Analysis</div>', unsafe_allow_html=True)

    weekly_hours = df.tail(7)["study_hours"].sum()
    burn_report  = burnout_analysis(
        burnout_score=burnout_score,
        avg_sleep=avg_sleep_7d,
        avg_stress=df.tail(14)["stress_level"].mean(),
        study_hours_week=weekly_hours,
    )

    color_map = {"Low": "#3FB950", "Moderate": "#D29922", "High": "#F78166", "Critical": "#FF0000"}
    bc = color_map[burn_report["level"]]

    b1, b2 = st.columns([1, 2])
    with b1:
        st.markdown(f"""
        <div class="metric-card" style="border-color:{bc}; padding: 30px;">
            <div class="metric-val" style="color:{bc}; font-size:3rem">{burn_report['score']}%</div>
            <div class="metric-label" style="color:{bc}; font-size:1rem">{burn_report['level']} Burnout Risk</div>
            <div style="margin-top:12px; font-size:0.85rem; color:#C9D1D9">{burn_report['message']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown(f"""
        <div class="badge-card">
            <div class="badge-emoji">{gam['badge']}</div>
            <div class="badge-title">{gam['rank']}</div>
            <div style="font-family:'JetBrains Mono'; color:#58A6FF; font-size:1.3rem">{gam['score']} pts</div>
            <div style="color:#8B949E; font-size:0.8rem">🔥 {gam['streak']}-day streak</div>
            <div style="color:#8B949E; font-size:0.75rem; margin-top:4px">{gam['next_rank_points']} pts to next rank</div>
        </div>
        """, unsafe_allow_html=True)

    with b2:
        fig_burn = plot_burnout_timeline(df)
        st.pyplot(fig_burn, use_container_width=True)

        st.markdown('<div class="section-header">💡 Recovery Actions</div>', unsafe_allow_html=True)
        for action in burn_report["actions"]:
            st.markdown(f"✅ {action}")

        st.markdown('<div class="section-header">📊 Wellness Stats</div>', unsafe_allow_html=True)
        wc1, wc2, wc3 = st.columns(3)
        wstats = [
            (wc1, f"{burn_report['sleep_deficit']}h", "Sleep Deficit"),
            (wc2, f"{burn_report['stress_rating']}", "Avg Stress (14d)"),
            (wc3, f"{burn_report['weekly_study_hours']}h", "Study Hrs This Week"),
        ]
        for col, val, lbl in wstats:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="font-size:1.3rem">{val}</div>
                    <div class="metric-label">{lbl}</div>
                </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8B949E; font-size:0.8rem; padding: 8px 0;">
    ⚡ StudyPulse – Smart Study Time Predictor &nbsp;|&nbsp; Built with Python · Scikit-Learn · Streamlit
</div>
""", unsafe_allow_html=True)
