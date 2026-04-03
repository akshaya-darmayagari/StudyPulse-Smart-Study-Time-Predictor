"""
StudyPulse – Synthetic Dataset Generator
Generates a realistic 90-day student study log with behavioral patterns.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
np.random.seed(SEED)

SUBJECTS = ["Mathematics", "Physics", "Chemistry", "Biology", "History", "English"]
DIFFICULTY = {"Mathematics": 9, "Physics": 8, "Chemistry": 7, "Biology": 6, "History": 4, "English": 3}

# Exam schedule (subject -> days from start)
EXAM_SCHEDULE = {
    "Mathematics": 30, "Physics": 45, "Chemistry": 60,
    "Biology": 50, "History": 75, "English": 80,
}

def productivity_curve(hour: int) -> float:
    """Returns a base productivity multiplier (0-1) for a given hour."""
    # Morning peak ~9-11, afternoon dip ~14-16, evening rise ~19-21
    morning  = 0.85 * np.exp(-((hour - 10) ** 2) / 4)
    afternoon = 0.45 * np.exp(-((hour - 15) ** 2) / 3)
    evening  = 0.70 * np.exp(-((hour - 20) ** 2) / 5)
    return float(np.clip(morning + afternoon + evening, 0.05, 1.0))


def generate_dataset(n_days: int = 90, sessions_per_day: int = 3) -> pd.DataFrame:
    start_date = datetime(2025, 6, 1)
    records = []

    # Student's "natural" chronotype: 0=morning, 1=neutral, 2=evening
    chronotype = np.random.choice([0, 1, 2])
    chronotype_offset = {0: -1, 1: 0, 2: 2}[chronotype]

    cumulative_stress = 0.0  # builds over time

    for day in range(n_days):
        date = start_date + timedelta(days=day)
        sleep_hours = float(np.clip(np.random.normal(6.8, 1.0), 4.0, 9.0))
        week_day = date.weekday()  # 0=Mon … 6=Sun
        is_weekend = week_day >= 5

        # Daily energy budget
        energy = (sleep_hours / 8.0) * 0.7 + 0.3 * np.random.random()
        cumulative_stress = float(np.clip(cumulative_stress + np.random.normal(0.02, 0.05), 0, 1))

        for session in range(sessions_per_day if not is_weekend else 2):
            subject = np.random.choice(SUBJECTS)
            difficulty = DIFFICULTY[subject] / 10.0

            # Days until subject exam
            days_to_exam = EXAM_SCHEDULE[subject] - day
            exam_pressure = float(np.clip(1 - days_to_exam / 30, 0, 1)) if days_to_exam > 0 else 0.9

            # Time-of-day selection
            hour_options = [8, 9, 10, 11, 14, 15, 16, 19, 20, 21]
            hour = int(np.random.choice(hour_options)) + chronotype_offset
            hour = int(np.clip(hour, 6, 23))

            base_prod = productivity_curve(hour)
            focus_level = float(np.clip(
                base_prod * energy * 10
                - difficulty * 1.5
                + exam_pressure * 1.5
                - cumulative_stress * 2.0
                + np.random.normal(0, 0.8),
                1, 10
            ))

            study_hours = float(np.clip(
                np.random.normal(1.5 + exam_pressure * 0.8, 0.4) * energy,
                0.25, 4.0
            ))

            break_duration = float(np.clip(
                study_hours * 0.25 * (1 + difficulty),
                5, 45
            ))  # in minutes

            # Previous score degrades with stress, improves with focus & study hours
            prev_score = float(np.clip(
                65 + focus_level * 2.5 + study_hours * 1.2
                - cumulative_stress * 10 + np.random.normal(0, 3),
                30, 100
            ))

            stress_level = float(np.clip(
                cumulative_stress * 10 + exam_pressure * 3 + difficulty * 2
                - sleep_hours * 0.5 + np.random.normal(0, 0.5),
                1, 10
            ))

            burnout_risk = float(np.clip(
                (cumulative_stress * 0.4 + (10 - sleep_hours) / 10 * 0.3
                 + stress_level / 10 * 0.3),
                0, 1
            ))

            time_of_day = (
                "Morning" if 5 <= hour < 12
                else "Afternoon" if 12 <= hour < 17
                else "Evening" if 17 <= hour < 21
                else "Night"
            )

            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "day_of_week": date.strftime("%A"),
                "hour": hour,
                "time_of_day": time_of_day,
                "subject": subject,
                "subject_difficulty": DIFFICULTY[subject],
                "study_hours": round(study_hours, 2),
                "focus_level": round(focus_level, 1),
                "sleep_hours": round(sleep_hours, 1),
                "break_duration_min": round(break_duration, 1),
                "days_to_exam": max(days_to_exam, 0),
                "exam_pressure": round(exam_pressure, 2),
                "previous_score": round(prev_score, 1),
                "stress_level": round(stress_level, 1),
                "burnout_risk": round(burnout_risk, 3),
                "is_weekend": int(is_weekend),
                "cumulative_stress": round(cumulative_stress, 3),
                "energy_level": round(energy, 3),
            })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out = "student_study_log.csv"
    df.to_csv(out, index=False)
    print(f"Dataset saved → {out}  |  shape: {df.shape}")
    print(df.head(3).to_string())
