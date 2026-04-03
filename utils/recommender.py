"""
StudyPulse – Recommendation & Schedule Engine
Generates personalised daily schedules, break suggestions,
exam prep strategies, and AI study method tips.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


SUBJECTS = ["Mathematics", "Physics", "Chemistry", "Biology", "History", "English"]
DIFFICULTY = {"Mathematics": 9, "Physics": 8, "Chemistry": 7, "Biology": 6, "History": 4, "English": 3}

STUDY_METHODS = {
    "Mathematics":  ["Active Problem Solving", "Spaced Repetition Drills", "Worked Examples Review"],
    "Physics":      ["Concept Mapping", "Formula Derivation Practice", "Past Paper Analysis"],
    "Chemistry":    ["Flashcard Memorisation", "Reaction Mechanism Diagrams", "Lab Report Review"],
    "Biology":      ["Mind Mapping", "Cornell Note Method", "Diagram Labelling"],
    "History":      ["Timeline Construction", "Essay Outlining", "Source Analysis"],
    "English":      ["Active Reading Annotation", "Vocabulary in Context", "Comparative Essay Practice"],
}

POMODORO_VARIANTS = {
    "high_focus":   (50, 10),   # (work_min, break_min)
    "medium_focus": (35, 10),
    "low_focus":    (25, 10),
    "burnout_risk": (20, 10),
}


# ── Productivity Curve ────────────────────────────────────────────────────────

def hourly_productivity_scores(base_sleep: float = 7.0, stress: float = 3.0) -> Dict[int, float]:
    """Return estimated focus score (0-10) for each hour 6-23."""
    scores = {}
    for h in range(6, 24):
        morning   = 0.85 * np.exp(-((h - 10) ** 2) / 4)
        afternoon = 0.45 * np.exp(-((h - 15) ** 2) / 3)
        evening   = 0.70 * np.exp(-((h - 20) ** 2) / 5)
        base = (morning + afternoon + evening) * 10
        sleep_bonus  = (base_sleep - 6) * 0.4
        stress_penalty = stress * 0.3
        scores[h] = round(float(np.clip(base + sleep_bonus - stress_penalty, 1, 10)), 2)
    return scores


def top_k_hours(hourly_scores: Dict[int, float], k: int = 5) -> List[int]:
    return sorted(hourly_scores, key=hourly_scores.get, reverse=True)[:k]


# ── Schedule Generator ────────────────────────────────────────────────────────

def generate_daily_schedule(
    subjects: List[str],
    hours_available: float,
    focus_scores_by_hour: Dict[int, float],
    exam_dates: Dict[str, int],    # subject -> days until exam
    sleep_hours: float = 7.0,
    stress_level: float = 4.0,
    burnout_risk: float = 0.2,
) -> List[Dict]:
    """
    Returns a list of time-blocks for the day, ordered by hour.
    Each block: {hour, subject, duration_min, focus_score, pomodoro, method, notes}
    """
    schedule = []
    peak_hours = top_k_hours(focus_scores_by_hour, k=6)

    # Weight subjects by difficulty × exam urgency
    weights = {}
    for subj in subjects:
        days = exam_dates.get(subj, 999)
        urgency = np.clip(1 - days / 30, 0, 1) if days > 0 else 0.95
        weights[subj] = DIFFICULTY[subj] / 10.0 * 0.5 + urgency * 0.5

    total_weight = sum(weights.values())
    # Allocate study time proportionally
    allocation = {s: (w / total_weight) * hours_available * 60 for s, w in weights.items()}

    # Sort subjects: hard + urgent first → place in peak hours
    sorted_subjects = sorted(subjects, key=lambda s: weights[s], reverse=True)

    # Pomodoro style
    if burnout_risk > 0.6:
        pomo_key = "burnout_risk"
    elif stress_level < 3:
        pomo_key = "high_focus"
    elif stress_level < 6:
        pomo_key = "medium_focus"
    else:
        pomo_key = "low_focus"
    work_min, break_min = POMODORO_VARIANTS[pomo_key]

    used_hours = set()
    for i, subj in enumerate(sorted_subjects):
        target_min = allocation[subj]
        blocks_needed = max(1, round(target_min / work_min))
        hour_pool = [h for h in peak_hours if h not in used_hours]
        if not hour_pool:
            hour_pool = [h for h in sorted(focus_scores_by_hour, key=focus_scores_by_hour.get, reverse=True)
                         if h not in used_hours]

        for b in range(min(blocks_needed, len(hour_pool))):
            h = hour_pool[b]
            used_hours.add(h)
            fscore = focus_scores_by_hour.get(h, 5.0)
            method = STUDY_METHODS[subj][b % len(STUDY_METHODS[subj])]
            days_left = exam_dates.get(subj, 999)
            note = (
                f"⚠️ Exam in {days_left}d — focus on weak areas!" if days_left <= 7
                else f"📅 Exam in {days_left}d — steady revision" if days_left <= 21
                else "🔄 Long-term practice"
            )
            schedule.append({
                "hour": h,
                "subject": subj,
                "duration_min": work_min,
                "break_after_min": break_min,
                "focus_score": fscore,
                "pomodoro_style": pomo_key.replace("_", " ").title(),
                "method": method,
                "notes": note,
            })

    schedule.sort(key=lambda x: x["hour"])
    return schedule


# ── Exam Prep Engine ──────────────────────────────────────────────────────────

def exam_prep_strategy(
    subject: str,
    days_until_exam: int,
    previous_score: float,
    avg_focus: float,
) -> Dict:
    """Returns an adaptive exam preparation plan."""
    difficulty = DIFFICULTY[subject]

    if days_until_exam <= 3:
        phase = "Final Sprint"
        daily_hours = 4.0
        focus_areas = ["Past Papers", "Formula Sheet Review", "Weak Topic Drills"]
        revision_pct = 90
    elif days_until_exam <= 7:
        phase = "Intensive Revision"
        daily_hours = 3.0
        focus_areas = ["Chapter Summaries", "Mock Tests", "Error Analysis"]
        revision_pct = 70
    elif days_until_exam <= 21:
        phase = "Active Practice"
        daily_hours = 2.0
        focus_areas = ["Concept Reinforcement", "Practice Problems", "Spaced Repetition"]
        revision_pct = 50
    else:
        phase = "Foundation Building"
        daily_hours = 1.5
        focus_areas = ["New Concepts", "Note Taking", "Light Review"]
        revision_pct = 20

    # Adjust for score
    if previous_score < 50:
        daily_hours *= 1.3
        focus_areas = ["Fundamentals Review"] + focus_areas
    elif previous_score > 80:
        daily_hours *= 0.85

    performance_tier = (
        "Needs Improvement" if previous_score < 50
        else "Average" if previous_score < 70
        else "Good" if previous_score < 85
        else "Excellent"
    )

    return {
        "subject": subject,
        "phase": phase,
        "days_until_exam": days_until_exam,
        "recommended_daily_hours": round(daily_hours, 1),
        "focus_areas": focus_areas,
        "revision_percentage": revision_pct,
        "new_content_percentage": 100 - revision_pct,
        "performance_tier": performance_tier,
        "previous_score": previous_score,
        "primary_method": STUDY_METHODS[subject][0],
    }


# ── Burnout Risk Analysis ─────────────────────────────────────────────────────

def burnout_analysis(
    burnout_score: float,
    avg_sleep: float,
    avg_stress: float,
    study_hours_week: float,
) -> Dict:
    if burnout_score < 0.25:
        level = "Low"
        colour = "green"
        message = "You're in great shape! Keep your current routine."
        actions = ["Maintain sleep schedule", "Continue regular breaks"]
    elif burnout_score < 0.50:
        level = "Moderate"
        colour = "yellow"
        message = "Some strain detected. Minor adjustments recommended."
        actions = ["Add 15-min mindfulness daily", "Take one full rest day/week",
                   "Reduce screen time 1hr before bed"]
    elif burnout_score < 0.75:
        level = "High"
        colour = "orange"
        message = "High burnout risk! Immediate lifestyle changes needed."
        actions = ["Reduce daily study by 1hr", "Sleep 7-8h strictly",
                   "Exercise 20min/day", "Limit difficult subjects to 2/day"]
    else:
        level = "Critical"
        colour = "red"
        message = "Critical burnout risk! Take a recovery break."
        actions = ["Take 1-2 day complete rest", "Consult a counsellor",
                   "Switch to light review only", "Prioritise sleep above all"]

    return {
        "level": level,
        "score": round(burnout_score * 100, 1),
        "colour": colour,
        "message": message,
        "actions": actions,
        "sleep_deficit": round(max(0, 7.5 - avg_sleep), 1),
        "stress_rating": round(avg_stress, 1),
        "weekly_study_hours": round(study_hours_week, 1),
    }


# ── Gamification Score ────────────────────────────────────────────────────────

def gamification_score(
    streak_days: int,
    avg_focus: float,
    sessions_completed: int,
    burnout_score: float,
) -> Dict:
    base_score = (
        streak_days * 10
        + avg_focus * 8
        + sessions_completed * 5
        - burnout_score * 50
    )
    score = max(0, min(1000, int(base_score)))

    if score < 200:   rank, badge = "Rookie",       "🌱"
    elif score < 400: rank, badge = "Learner",      "📖"
    elif score < 600: rank, badge = "Scholar",      "🎓"
    elif score < 800: rank, badge = "Achiever",     "⭐"
    else:             rank, badge = "Study Master", "🏆"

    return {
        "score": score,
        "rank": rank,
        "badge": badge,
        "streak": streak_days,
        "next_rank_points": 200 - (score % 200),
    }
