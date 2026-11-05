# Data dictionary

## `raw/survey_responses.csv` (Google Forms export, timestamps removed)

| Column | Question (translated from Thai) |
|---|---|
| `response_id` | Row id |
| `age_group` | Q1 Current age: <10, 10–12, 13–15, 16–18, 19+ |
| `favourite_activity` | Q2 Favourite offline recreational activity (choose one / free text) |
| `q3_enjoyment_1_10` | Q3 Rating of the fun and appeal of that activity (1–10) |
| `q4_weather` | Q4 Weather during the past week when usually doing the activity. Favourable → 1, unfavourable (very hot / rain / PM2.5) → 0 |
| `q5_friends` | Q5 Number of friends who usually join |
| `q6_parent_support_1_5` | Q6 Parental support: cost, transport, permission (1–5) |
| `q7_screen_entertain_h_day` | Q7 Average daily entertainment screen time: games, series, social media (hours) |
| `q8_screen_study_h_day` | Q8 Average daily study or homework screen time (hours) |
| `q9_offscreen_min_week` | Q9 Total offline sport and recreation time in the last 7 days (minutes) |

## `processed/survey_coded.csv`

| Column | Meaning |
|---|---|
| `Y_offscreen_min_week` | Q9 |
| `log_Y` | ln(1 + Y), the response in the main model |
| `X1_enjoyment` … `X6_screen_study_h` | Q3, Q4 (dummy), Q5, Q6, Q7, Q8 |
| `age_group`, `activity` | English labels for Q1, Q2 |
| `gen_alpha` | 1 if age band is <10, 10–12, 13–15 or 16–18 (born 2010+ ≈ age ≤ 16 in 2026; the single 16–18 respondent is kept), 0 for 19+ |
