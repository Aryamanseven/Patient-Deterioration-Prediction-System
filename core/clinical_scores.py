"""
Clinical Early Warning Scores — NEWS, MEWS, qSOFA.

These are standard clinical scoring systems used in hospitals worldwide
to assess patient deterioration risk. We use them in two ways:
1. As FEATURES for our ML models (giving them clinical domain knowledge)
2. As BASELINES to compare against (proving our AI outperforms them)

References:
- NEWS: Royal College of Physicians (2017). National Early Warning Score 2
- MEWS: Modified Early Warning Score (Subbe et al., 2001)
- qSOFA: Quick Sequential Organ Failure Assessment (Singer et al., 2016)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_news(row: pd.Series) -> int:
    """
    National Early Warning Score (NEWS).
    Ranges from 0-20. Score >= 5 indicates clinical concern.

    Parameters based on Royal College of Physicians 2017 standard:
    - Respiratory Rate
    - SpO2
    - Systolic BP
    - Heart Rate
    - Temperature
    - Consciousness (approximated by mobility_score)
    """
    score = 0

    # Respiratory rate
    rr = row.get("respiratory_rate", 16)
    if rr <= 8:
        score += 3
    elif rr <= 11:
        score += 1
    elif rr <= 20:
        score += 0
    elif rr <= 24:
        score += 2
    else:
        score += 3

    # SpO2
    spo2 = row.get("spo2_pct", 97)
    if spo2 <= 91:
        score += 3
    elif spo2 <= 93:
        score += 2
    elif spo2 <= 95:
        score += 1

    # Systolic BP
    sbp = row.get("systolic_bp", 120)
    if sbp <= 90:
        score += 3
    elif sbp <= 100:
        score += 2
    elif sbp <= 110:
        score += 1
    elif sbp <= 219:
        score += 0
    else:
        score += 3

    # Heart rate
    hr = row.get("heart_rate", 75)
    if hr <= 40:
        score += 3
    elif hr <= 50:
        score += 1
    elif hr <= 90:
        score += 0
    elif hr <= 110:
        score += 1
    elif hr <= 130:
        score += 2
    else:
        score += 3

    # Temperature
    temp = row.get("temperature_c", 37.0)
    if temp <= 35.0:
        score += 3
    elif temp <= 36.0:
        score += 1
    elif temp <= 38.0:
        score += 0
    elif temp <= 39.0:
        score += 1
    else:
        score += 2

    # Supplemental oxygen (approximated via oxygen_flow)
    o2_flow = row.get("oxygen_flow", 0)
    if o2_flow > 0:
        score += 2

    # Consciousness (approximated: mobility 0-1 = impaired)
    mobility = row.get("mobility_score", 3)
    if mobility <= 1:
        score += 3

    return score


def compute_mews(row: pd.Series) -> int:
    """
    Modified Early Warning Score (MEWS).
    Ranges from 0-14. Score >= 4 triggers medical review.

    Based on Subbe et al. (2001).
    """
    score = 0

    # Systolic BP
    sbp = row.get("systolic_bp", 120)
    if sbp <= 70:
        score += 3
    elif sbp <= 80:
        score += 2
    elif sbp <= 100:
        score += 1
    elif sbp <= 199:
        score += 0
    else:
        score += 2

    # Heart rate
    hr = row.get("heart_rate", 75)
    if hr <= 40:
        score += 2
    elif hr <= 50:
        score += 1
    elif hr <= 100:
        score += 0
    elif hr <= 110:
        score += 1
    elif hr <= 129:
        score += 2
    else:
        score += 3

    # Respiratory rate
    rr = row.get("respiratory_rate", 16)
    if rr < 9:
        score += 2
    elif rr <= 14:
        score += 0
    elif rr <= 20:
        score += 1
    elif rr <= 29:
        score += 2
    else:
        score += 3

    # Temperature
    temp = row.get("temperature_c", 37.0)
    if temp < 35.0:
        score += 2
    elif temp <= 38.4:
        score += 0
    else:
        score += 2

    # Consciousness (mobility proxy)
    mobility = row.get("mobility_score", 3)
    if mobility <= 1:
        score += 3
    elif mobility == 2:
        score += 1

    return score


def compute_qsofa(row: pd.Series) -> int:
    """
    Quick Sequential Organ Failure Assessment (qSOFA).
    Ranges from 0-3. Score >= 2 suggests possible sepsis.

    Based on Singer et al. (2016), JAMA.
    """
    score = 0

    # Respiratory rate >= 22
    if row.get("respiratory_rate", 16) >= 22:
        score += 1

    # Altered mentation (mobility <= 1 as proxy)
    if row.get("mobility_score", 3) <= 1:
        score += 1

    # Systolic BP <= 100
    if row.get("systolic_bp", 120) <= 100:
        score += 1

    return score


def compute_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all clinical scores and add as columns."""
    result = df.copy()
    result["news_score"] = df.apply(compute_news, axis=1).astype("float32")
    result["mews_score"] = df.apply(compute_mews, axis=1).astype("float32")
    result["qsofa_score"] = df.apply(compute_qsofa, axis=1).astype("float32")
    return result


def compute_scores_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized computation of clinical scores (faster than row-wise apply)."""
    result = df.copy()

    # NEWS — vectorized
    news = np.zeros(len(df), dtype="float32")

    rr = df["respiratory_rate"].values
    news += np.where(rr <= 8, 3, np.where(rr <= 11, 1, np.where(rr <= 20, 0, np.where(rr <= 24, 2, 3))))

    spo2 = df["spo2_pct"].values
    news += np.where(spo2 <= 91, 3, np.where(spo2 <= 93, 2, np.where(spo2 <= 95, 1, 0)))

    sbp = df["systolic_bp"].values
    news += np.where(sbp <= 90, 3, np.where(sbp <= 100, 2, np.where(sbp <= 110, 1, np.where(sbp <= 219, 0, 3))))

    hr = df["heart_rate"].values
    news += np.where(hr <= 40, 3, np.where(hr <= 50, 1, np.where(hr <= 90, 0, np.where(hr <= 110, 1, np.where(hr <= 130, 2, 3)))))

    temp = df["temperature_c"].values
    news += np.where(temp <= 35.0, 3, np.where(temp <= 36.0, 1, np.where(temp <= 38.0, 0, np.where(temp <= 39.0, 1, 2))))

    o2 = df["oxygen_flow"].values
    news += np.where(o2 > 0, 2, 0)

    mob = df["mobility_score"].values
    news += np.where(mob <= 1, 3, 0)

    result["news_score"] = news

    # MEWS — vectorized
    mews = np.zeros(len(df), dtype="float32")
    mews += np.where(sbp <= 70, 3, np.where(sbp <= 80, 2, np.where(sbp <= 100, 1, np.where(sbp <= 199, 0, 2))))
    mews += np.where(hr <= 40, 2, np.where(hr <= 50, 1, np.where(hr <= 100, 0, np.where(hr <= 110, 1, np.where(hr <= 129, 2, 3)))))
    mews += np.where(rr < 9, 2, np.where(rr <= 14, 0, np.where(rr <= 20, 1, np.where(rr <= 29, 2, 3))))
    mews += np.where(temp < 35.0, 2, np.where(temp <= 38.4, 0, 2))
    mews += np.where(mob <= 1, 3, np.where(mob == 2, 1, 0))
    result["mews_score"] = mews

    # qSOFA — vectorized
    qsofa = np.zeros(len(df), dtype="float32")
    qsofa += np.where(rr >= 22, 1, 0)
    qsofa += np.where(mob <= 1, 1, 0)
    qsofa += np.where(sbp <= 100, 1, 0)
    result["qsofa_score"] = qsofa

    return result
