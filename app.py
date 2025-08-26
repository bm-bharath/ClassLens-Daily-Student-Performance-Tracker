# app.py
import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------- Page setup ----------------
st.set_page_config(page_title="Student Performance Daily", page_icon="📊", layout="wide")
st.title("📊 Student Daily Performance Dashboard")

# Sidebar feature toggles (optional)
USE_AI = st.sidebar.toggle("Enable AI insights", value=False, help="Optional AI-generated coach notes")
USE_FEEDBACK = st.sidebar.toggle("Enable feedback widget", value=False, help="Collect user feedback in-app")

# Lazy import for optional features
if USE_AI:
    try:
        import openai
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
        else:
            st.info("Set OPENAI_API_KEY in Streamlit secrets or environment to enable AI insights.")
    except Exception:
        OPENAI_API_KEY = None
        st.info("Install openai to enable AI insights: pip install openai")

if USE_FEEDBACK:
    try:
        from streamlit_feedback import streamlit_feedback
    except Exception:
        streamlit_feedback = None

# ---------------- Data expectations ----------------
with st.expander("Data format expectations", expanded=False):
    st.markdown("""
    Required columns (case-insensitive):
    - date (YYYY-MM-DD)
    - register_no
    - student_name
    - present: P (present) or A (absent)
    - set: Set1, Set2, ... Include only sets that actually occurred that day
    - q1–q4: Yes (correct), No (wrong), a or blank (not attempted)
    - q5_rating: 1–5 (feedback) or blank

    One row = one student × one test set × one date.
    Some days may have only Set1; others may have Set1 and Set2. Add rows only for sets that happened.
    """)

# Sample CSV template (downloadable)
sample = pd.DataFrame({
    "date": ["2025-07-19","2025-07-19","2025-07-20","2025-07-21","2025-07-21"],
    "register_no": ["9923001001","9923001001","9923001002","9923001005","9923001005"],
    "student_name": ["PADALA SUSHANTH","PADALA SUSHANTH","PRIYAN S.V","THOTA UDAYSRI","THOTA UDAYSRI"],
    "present": ["P","P","A","P","P"],
    "set": ["Set1","Set2","Set1","Set1","Set2"],
    "q1": ["Yes","yes","a","Yes","yes"],
    "q2": ["Yes","yes","a","Yes","no"],
    "q3": ["yes","yes","a","Yes","yes"],
    "q4": ["no","yes","a","Yes","yes"],
    "q5_rating": [5,5,"",4,4],
})
st.download_button("Download CSV template", sample.to_csv(index=False).encode(), file_name="student_daily_template.csv", mime="text/csv")  # [1]

uploaded = st.file_uploader("Upload daily CSV", type=["csv"])
if uploaded is None:
    st.stop()

# ---------------- Load and clean ----------------
df = pd.read_csv(uploaded)
df.columns = [c.strip().lower() for c in df.columns]

required = {"date","register_no","student_name","present","set","q1","q2","q3","q4","q5_rating"}
missing = required - set(df.columns)
if missing:
    st.error(f"Missing required columns: {sorted(list(missing))}")
    st.stop()

# Attendance: P/A -> True/False
def parse_present(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in ["P","p","present","yes","y","1","true","t"]: return True
    if s in ["A","a","absent","no","n","0","false","f"]: return False
    return np.nan

# Questions: Yes/No/a -> True/False/NaN
def parse_q(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in ["yes","y","true","t","1"]: return True
    if s in ["no","n","false","f","0"]: return False
    if s in ["a","absent","na","n/a","-",""]:
        return np.nan
    return np.nan

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["present"] = df["present"].apply(parse_present)
for q in ["q1","q2","q3","q4"]:
    df[q] = df[q].apply(parse_q)
df["q5_rating"] = pd.to_numeric(df["q5_rating"], errors="coerce")

# Drop rows missing critical fields
core_missing = df["date"].isna() | df["register_no"].isna() | df["student_name"].isna() | df["present"].isna() | df["set"].isna()
dropped = int(core_missing.sum())
if dropped:
    st.warning(f"Dropped {dropped} rows with invalid core fields (date/register_no/student_name/present/set).")  # [1]
df = df[~core_missing].copy()

# Per-row metrics (ignore NaNs in accuracy)
df["question_count"] = df[["q1","q2","q3","q4"]].notna().sum(axis=1)
df["correct_count"] = df[["q1","q2","q3","q4"]].apply(lambda r: int(np.nansum(r.astype(float))) if r.notna().any() else 0, axis=1)
df["accuracy"] = np.where(df["question_count"]>0, df["correct_count"]/df["question_count"], np.nan)

# ---------------- Filters ----------------
with st.sidebar:
    students = sorted(df["student_name"].dropna().unique().tolist())
    selected_students = st.multiselect("Students", students, default=students)
    sets = sorted(df["set"].dropna().unique().tolist())
    selected_sets = st.multiselect("Sets", sets, default=sets)
    dmin, dmax = df["date"].min().date(), df["date"].max().date()
    
    min_rating = st.slider("Min feedback rating (Q5)", 1, 5, 1)
    # After computing dmin, dmax:
    date_selection = st.date_input("Date range", value=(dmin, dmax))

# Normalize to start/end dates, whether a single date or a range was returned
    if isinstance(date_selection, tuple) or isinstance(date_selection, list):
        if len(date_selection) == 2 and all(date_selection):
            start_date, end_date = date_selection
        elif len(date_selection) == 1 and date_selection is not None:
            start_date = end_date = date_selection
        else:
        # empty selection -> use full span
            start_date, end_date = dmin, dmax
    else:
    # single date
        start_date = end_date = date_selection

# Build mask using the two bounds
    mask = (
        df["student_name"].isin(selected_students) &
        df["set"].isin(selected_sets) &
        (df["date"].dt.date >= start_date) &
        (df["date"].dt.date <= end_date) &
        (df["q5_rating"].fillna(0) >= min_rating)
    )
    view = df.loc[mask].copy()


# ---------------- KPIs ----------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Records", len(view))
with col2:
    # Accurate attendance: include all rows in the date/student/set window
    date_mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
    stu_mask  = df["student_name"].isin(selected_students)
    set_mask  = df["set"].isin(selected_sets)

    # 1-D Series of booleans
    att_base = df.loc[date_mask & stu_mask & set_mask, "present"]
    total_rows = att_base.shape[0]
    present_rows = int((att_base == True).sum())
    att_rate = (present_rows / total_rows) if total_rows > 0 else np.nan
    st.metric("Attendance rate", f"{att_rate*100:.1f}%" if total_rows > 0 else "—")


with col3:
    st.metric("Avg accuracy", f"{(view['accuracy'].mean()*100):.1f}%" if view["accuracy"].notna().any() else "—")
with col4:
    st.metric("Avg rating (Q5)", f"{view['q5_rating'].mean():.2f}" if view["q5_rating"].notna().any() else "—")

# ---------------- Tabs: Overview vs Individual ----------------
st.header("Analysis")
tab_overall, tab_individual = st.tabs(["Overview", "Individual"])  # [8]

with tab_overall:
    # Per-student aggregates
    agg = (
        view.groupby(["register_no","student_name"], as_index=False)
            .agg(
                days_present=("present","sum"),
                days_total=("present","count"),
                tests_taken=("set","count"),
                q_total=("question_count","sum"),
                q_correct=("correct_count","sum"),
                accuracy_mean=("accuracy","mean"),
                rating_mean=("q5_rating","mean"),
            )
    )
    agg["attendance_rate"] = np.where(agg["days_total"]>0, agg["days_present"]/agg["days_total"], np.nan)
    agg["accuracy_overall"] = np.where(agg["q_total"]>0, agg["q_correct"]/agg["q_total"], np.nan)

    st.subheader("Per-student summary")
    st.dataframe(
        agg[["register_no","student_name","attendance_rate","accuracy_overall","rating_mean","tests_taken","days_total"]]
          .sort_values(["accuracy_overall","attendance_rate"], ascending=[False, False])
          .style.format({"attendance_rate":"{:.1%}","accuracy_overall":"{:.1%}","rating_mean":"{:.2f}"}),
        use_container_width=True
    )

    # Daily accuracy (mean across sets)
    daily = (
        view.groupby(["date","student_name"], as_index=False)
            .agg(acc=("accuracy","mean"), present=("present","max"), rating=("q5_rating","mean"))
    )
    st.caption("Shows mean accuracy per student per day; each point averages all sets taken that date. Higher, stable lines indicate consistency.")  # [web:1]
    fig = px.line(
        daily.sort_values("date"),
        x="date", y="acc", color="student_name", markers=True,
        title="Daily accuracy (mean across sets)",
    )
    fig.update_layout(legend_title_text="Student", yaxis_title="Accuracy", xaxis_title="Date")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


    # Set-wise accuracy
    by_set = (
        view.groupby(["student_name","set"], as_index=False)
            .agg(acc=("accuracy","mean"), n=("accuracy","count"))
    )
    st.caption("Compares each student’s average accuracy by Set1/Set2. Large gaps suggest set‑specific difficulty or timing effects.")  # [web:1]
    fig2 = px.bar(
        by_set, x="student_name", y="acc", color="set", barmode="group",
        hover_data=["n"], title="Set-wise accuracy (mean)"
    )
    fig2.update_layout(xaxis_tickangle=-45, legend_title_text="Set", yaxis_title="Accuracy")
    fig2.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)


    # Attendance vs accuracy
    acc_att = (
        view.groupby(["student_name"], as_index=False)
            .agg(attendance=("present","mean"), acc=("accuracy","mean"))
    )
    st.caption("Each dot is a student; x-axis = attendance rate, y-axis = mean accuracy. The trendline shows the relationship between attendance and performance.")  # [web:1]
    fig3 = px.scatter(
        acc_att, x="attendance", y="acc", color="student_name",
        title="Attendance vs Accuracy", trendline="ols"
    )
    fig3.update_layout(legend_title_text="Student", xaxis_title="Attendance", yaxis_title="Accuracy")
    fig3.update_xaxes(tickformat=".0%")
    fig3.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig3, use_container_width=True)


with tab_individual:
    # Student selector within the filtered view
    if view.empty:
        st.info("No records for the current filters.")
    else:
        student_pick = st.selectbox("Choose a student", options=sorted(view["student_name"].unique()))
        sview = view[view["student_name"]==student_pick].sort_values(["date","set"]).copy()

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Days present", int(sview["present"].sum()))
        with c2: st.metric("Tests taken", int(sview["set"].count()))
        with c3: st.metric("Mean accuracy", f"{sview['accuracy'].mean()*100:.1f}%")
        with c4: st.metric("Mean rating", f"{sview['q5_rating'].mean():.2f}" if sview["q5_rating"].notna().any() else "—")

        # Moving averages
        s_daily = (sview.groupby("date", as_index=False)
                        .agg(acc=("accuracy","mean"), present=("present","max"), rating=("q5_rating","mean"))
                        .sort_values("date"))
        s_daily["acc_7d"] = s_daily["acc"].rolling(7, min_periods=1).mean()
        s_daily["acc_14d"] = s_daily["acc"].rolling(14, min_periods=1).mean()
        fig_s = px.line(s_daily, x="date", y=["acc","acc_7d","acc_14d"],
                        markers=True, title=f"{student_pick}: accuracy and moving averages")
        st.plotly_chart(fig_s, use_container_width=True)

        # Set-wise accuracy and participation
        by_set_s = sview.groupby("set", as_index=False).agg(
            acc=("accuracy","mean"),
            n=("accuracy","count"),
            answered_q=("question_count","sum"),
            correct_q=("correct_count","sum"),
        )
        fig_set = px.bar(by_set_s, x="set", y="acc", text="n", title="Set-wise mean accuracy (n = tests)")
        st.plotly_chart(fig_set, use_container_width=True)

        # Per-question mastery and attempt rates
        qcols = ["q1","q2","q3","q4"]
        mastery = pd.DataFrame({
            "question": qcols,
            "correct_rate": [(sview[q]==True).mean() if sview[q].notna().any() else np.nan for q in qcols],
            "attempt_rate": [sview[q].notna().mean() for q in qcols],
        })
        col_a, col_b = st.columns(2)
        with col_a:
            fig_q = px.bar(mastery, x="question", y="correct_rate", range_y=[0,1], title="Per-question correct rate")
            st.plotly_chart(fig_q, use_container_width=True)
        with col_b:
            fig_a = px.bar(mastery, x="question", y="attempt_rate", range_y=[0,1], title="Per-question attempt rate")
            st.plotly_chart(fig_a, use_container_width=True)

        # Attendance effect
        acc_att_s = (sview.groupby("present", as_index=False)
                           .agg(acc=("accuracy","mean"), n=("accuracy","count")))
        acc_att_s["status"] = acc_att_s["present"].map({True:"Present rows", False:"Absent rows"})
        fig_att = px.bar(acc_att_s, x="status", y="acc", text="n", title="Accuracy by attendance status")
        st.plotly_chart(fig_att, use_container_width=True)

        # Detailed table with ✅/❌/—
        qtbl = sview[["date","set"]+qcols+["q5_rating","accuracy"]].copy()
        qtbl[qcols] = qtbl[qcols].replace({True:"✅", False:"❌"}).fillna("—")
        qtbl["accuracy"] = (qtbl["accuracy"]*100).round(1)
        st.dataframe(qtbl.sort_values(["date","set"]), use_container_width=True)

        # Optional AI coach notes
        if USE_AI and 'openai' in globals() and OPENAI_API_KEY:
            meta = {
                "dates": s_daily["date"].dt.strftime("%Y-%m-%d").tolist(),
                "acc": [float(x) for x in s_daily["acc"].round(3).tolist()],
                "acc_7d": [float(x) for x in s_daily["acc_7d"].round(3).tolist()],
                "acc_14d": [float(x) for x in s_daily["acc_14d"].round(3).tolist()],
                "set_acc": {k: float(v) for k,v in by_set_s.set_index("set")["acc"].to_dict().items()},
                "q_correct_rates": {row["question"]: (None if pd.isna(row["correct_rate"]) else float(row["correct_rate"]))
                                    for _, row in mastery.iterrows()},
                "attendance_mean": float(sview["present"].mean()),
                "rating_mean": (None if not sview["q5_rating"].notna().any() else float(sview["q5_rating"].mean())),
            }
            prompt = f"""Act as an educational coach. Using this student's metrics (JSON), provide:
- 2 strengths, 2 priority improvements,
- 3 concrete practice suggestions (reference sets/questions),
- 1 actionable step for tomorrow.
Be concise and evidence-based.
JSON:\n{json.dumps(meta, indent=2)}"""
            try:
                # Modern chat API; fallback if needed
                try:
                    resp = openai.chat.completions.create(
                        model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
                        messages=[{"role":"user","content":prompt}],
                        temperature=0.2,
                    )
                    text = resp.choices.message.content
                except Exception:
                    resp = openai.ChatCompletion.create(
                        model=os.getenv("OPENAI_MODEL","gpt-3.5-turbo"),
                        messages=[{"role":"user","content":prompt}],
                        temperature=0.2,
                    )
                    text = resp.choices.message["content"]
                st.subheader("AI coach notes")
                st.markdown(text)
            except Exception as e:
                st.info(f"AI disabled or error: {e}")

# ---------------- Feedback (optional) ----------------
if USE_FEEDBACK:
    st.subheader("Feedback")
    if 'streamlit_feedback' in globals() and streamlit_feedback:
        fb = streamlit_feedback(feedback_type="thumbs", optional_text_label="[Optional] Ideas to improve the dashboard")
        if fb:
            st.write("Thanks for the feedback!")
    else:
        st.info("Install streamlit-feedback: pip install streamlit-feedback")

# ---------------- Download ----------------
st.subheader("Download processed data")
buf = io.StringIO()
view.to_csv(buf, index=False)
st.download_button("Download filtered CSV", buf.getvalue(), file_name="filtered_student_daily.csv", mime="text/csv")
