"""
Raw Google Forms export -> coded, analysis-ready table.

    python src/clean.py

Rules (all decisions are flags, nothing is silently deleted):
  * Weather answer text -> dummy X2 (1 = favourable, 0 = unfavourable), as the form itself specifies.
  * Age band -> gen_alpha flag. Gen Alpha = born 2010 or later = age <= 16 in 2026.
    Bands "<10", "10-12", "13-15" qualify; "16-18" is kept (1 respondent, band overlaps 16);
    "19+" is excluded from the main analysis.
  * Y is right-skewed with zeros (0-1,680 min/week), so the main model uses log(1 + Y).
    This keeps every respondent instead of deleting extreme values.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

AGE = {"ต่ำกว่า 10 ปี": "<10", "10 - 12 ปี": "10-12", "13 - 15 ปี": "13-15",
       "16 - 18 ปี": "16-18", "19 ปี ขึ้นไป": "19+"}
ACTIVITY = {"กีฬาประเภททีม": "Team sport", "กีฬาประเภทเดี่ยว": "Individual sport",
            "กิจกรรมศิลปะ/ดนตรี": "Art / music", "กิจกรรมกลางแจ้ง/ผจญภัย": "Outdoor / adventure"}


def activity_en(text: str) -> str:
    for th, en in ACTIVITY.items():
        if str(text).startswith(th):
            return en
    return "Other"


def main():
    raw = pd.read_csv(ROOT / "data/raw/survey_responses.csv")
    df = pd.DataFrame({
        "response_id": raw.response_id,
        "age_group": raw.age_group.str.strip().map(AGE),
        "activity": raw.favourite_activity.map(activity_en),
        "Y_offscreen_min_week": raw.q9_offscreen_min_week.astype(float),
        "X1_enjoyment": raw.q3_enjoyment_1_10,
        "X2_good_weather": raw.q4_weather.str.contains("แปลงค่าเป็น 1").astype(int),
        "X3_friends": raw.q5_friends,
        "X4_parent_support": raw.q6_parent_support_1_5,
        "X5_screen_entertain_h": raw.q7_screen_entertain_h_day.astype(float),
        "X6_screen_study_h": raw.q8_screen_study_h_day.astype(float),
    })
    assert df.age_group.notna().all()
    df["gen_alpha"] = (df.age_group != "19+").astype(int)
    df["log_Y"] = np.log1p(df.Y_offscreen_min_week).round(5)
    df.to_csv(ROOT / "data/processed/survey_coded.csv", index=False)
    print(f"coded {len(df)} responses; Gen Alpha = {df.gen_alpha.sum()}")


if __name__ == "__main__":
    main()
