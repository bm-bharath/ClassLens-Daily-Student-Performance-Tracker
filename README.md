# ClassLens-Daily-Student-Performance-Tracker
Student tool to track learning each day. Upload a simple CSV to see who was present (P/A), how each student scored on daily quiz questions (Yes/No), and their 1–5 feedback rating. View class summaries and drill down to any student to spot strengths, gaps, and trends. Works even when some days have one test and others have two. No coding needed.

A lightweight Streamlit app to analyze daily attendance and quiz results (Yes/No for Q1–Q4; Q5 is a 1–5 feedback rating). Supports variable tests per day (Set1/Set2), P/A attendance, and per‑student drill‑downs.

## Features
- CSV upload with schema validation and cleaning (P/A; Yes/No/a).  
- Overview KPIs, per‑student summary, daily accuracy lines, set‑wise bars, attendance vs accuracy.  
- Individual tab: moving averages (7/14d), per‑question mastery, attendance effect, detailed table.  
- Optional AI coach notes and in‑app feedback (toggle in sidebar).

## Data format (CSV columns)
`date, register_no, student_name, present, set, q1, q2, q3, q4, q5_rating`  
- `present`: P=present, A=absent  
- `q1–q4`: Yes/No/`a` (not attempted)  
- `q5_rating`: 1–5 or blank

