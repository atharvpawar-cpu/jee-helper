import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
from datetime import date, datetime, timedelta
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# -------------------- CONFIG --------------------
st.set_page_config(page_title="JEE Prep Tracker", page_icon="📘", layout="wide")
DATA_FILE = "jee_data.csv"
TASKS_FILE = "tasks.csv"

# -------------------- THEME TOGGLE (charts only) --------------------
dark = st.toggle("🌙 Dark chart theme", value=True)
plt.style.use("dark_background" if dark else "default")

# -------------------- HELPERS --------------------
def load_csv(path: str, cols: list) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        for c in cols:
            if c not in df.columns:
                df[c] = [] if c not in ("Date",) else pd.NaT
        return df
    return pd.DataFrame(columns=cols)

def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    for c in ["Physics", "Chemistry", "Maths", "Total"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    if "StudyHours" in df.columns:
        df["StudyHours"] = pd.to_numeric(df["StudyHours"], errors="coerce").fillna(0.0)
    if "Student" in df.columns:
        df["Student"] = df["Student"].fillna("You").astype(str)
    return df

def estimate_percentile(total_score: int, total_max: int = 300) -> float:
    mean = 0.5 * total_max
    std = max(1.0, 0.20 * total_max)
    z = (total_score - mean) / std
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    d = 0.3989423 * np.exp(-z*z/2.0)
    prob = 1 - d*(1.330274*t - 1.821256*t**2 + 1.781478*t**3 - 0.356538*t**4 + 0.3193815*t**5)
    if z < 0:
        prob = 1 - prob
    return float(np.clip(prob*100, 0, 100))

def estimate_rank(percentile: float, candidates: int = 1_000_000) -> int:
    return max(1, int(round((100.0 - percentile) / 100.0 * candidates)) + 1)

def weakness_tip(subject: str) -> str:
    tips = {
        "Physics": "Focus on mechanics basics (vectors, NLM), revise formula sheet daily, and practice error analysis.",
        "Chemistry": "Revise NCERT line-by-line, especially Inorganic. Practice Organic mechanisms and P&C numericals.",
        "Maths": "Strengthen algebra & coordinate geometry basics. Drill standard problems and keep a ‘mistake log’.",
        "Balanced": "You’re fairly balanced. Push recent weak chapters and improve test strategy (time split & guessing).",
    }
    return tips.get(subject, "Revise fundamentals and solve timed mixed sets.")

def calc_streak(dates: list) -> int:
    if not dates:
        return 0
    s = sorted({d for d in dates if isinstance(d, date)}, reverse=True)
    if not s:
        return 0
    streak = 1 if s[0] == date.today() else 0
    cur = s[0]
    for d in s[1:]:
        if cur - d == timedelta(days=1):
            streak += 1
            cur = d
        else:
            break
    return streak

def calendar_heatmap(df: pd.DataFrame, value_col: str = "Total"):
    if df.empty:
        st.info("Add tests to see the calendar heatmap.")
        return
    dfx = df.copy()
    dfx["Date"] = pd.to_datetime(dfx["Date"])
    dfx["Week"] = dfx["Date"].dt.isocalendar().week.astype(int)
    dfx["Year"] = dfx["Date"].dt.year.astype(int)
    dfx["Weekday"] = dfx["Date"].dt.weekday  # 0=Mon ... 6=Sun
    latest_year = int(dfx["Year"].max())
    grid = dfx[dfx["Year"] == latest_year].pivot_table(
        index="Weekday", columns="Week", values=value_col, aggfunc="mean"
    )
    grid = grid.sort_index()  # 0..6
    fig = go.Figure(data=go.Heatmap(
        z=grid.values if grid.size else [[None]],
        x=[str(x) for x in grid.columns] if not grid.empty else ["-"],
        y=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        hoverongaps=False,
        coloraxis="coloraxis"
    ))
    fig.update_layout(
        title=f"Calendar Heatmap — {value_col} (Year {latest_year})",
        xaxis_title="ISO Week",
        yaxis_title="Weekday",
        coloraxis_colorbar=dict(title=value_col),
        template="plotly_dark" if dark else "plotly"
    )
    st.plotly_chart(fig, use_container_width=True)

def pdf_report(df: pd.DataFrame, fname: str = "JEE_Report.pdf"):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height-50, "JEE Prep Progress Report")

    c.setFont("Helvetica", 11)
    y = height - 90
    if df.empty:
        c.drawString(40, y, "No data available.")
    else:
        latest = df.iloc[-1]
        avgP = df["Physics"].mean(); bestP = df["Physics"].max()
        avgC = df["Chemistry"].mean(); bestC = df["Chemistry"].max()
        avgM = df["Maths"].mean(); bestM = df["Maths"].max()
        avgT = df["Total"].mean(); bestT = df["Total"].max()

        c.drawString(40, y, f"Latest: {latest['Date']} | P {latest['Physics']}  C {latest['Chemistry']}  M {latest['Maths']}  Total {latest['Total']}")
        y -= 20
        c.drawString(40, y, f"Averages: Physics {avgP:.1f}, Chemistry {avgC:.1f}, Maths {avgM:.1f}, Total {avgT:.1f}")
        y -= 20
        c.drawString(40, y, f"Bests:    Physics {bestP}, Chemistry {bestC}, Maths {bestM}, Total {bestT}")
        y -= 30

        perc = estimate_percentile(int(latest["Total"]))
        rank = estimate_rank(perc)
        c.drawString(40, y, f"Estimated Percentile: {perc:.2f}")
        y -= 20
        c.drawString(40, y, f"Estimated AIR: {rank}")
        y -= 30

        means = {
            "Physics": df["Physics"].tail(5).mean() if len(df) else 0,
            "Chemistry": df["Chemistry"].tail(5).mean() if len(df) else 0,
            "Maths": df["Maths"].tail(5).mean() if len(df) else 0,
        }
        weakest = min(means, key=means.get)
        tip = weakness_tip(weakest)
        c.drawString(40, y, f"Weakest Subject (recent): {weakest}")
        y -= 20
        c.drawString(40, y, f"Tip: {tip[:85]}")
        y -= 40

        c.setFont("Helvetica-Oblique", 9)
        c.drawString(40, y, "Note: Percentile & AIR are rough estimates; actual results vary by shift & normalization.")

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()

# -------------------- LOAD DATA --------------------
df = load_csv(DATA_FILE, ["Date","Student","Physics","Chemistry","Maths","Total","StudyHours"])
df = ensure_types(df)

tasks = load_csv(TASKS_FILE, ["Date","Task","Done"])
if not tasks.empty:
    tasks["Date"] = pd.to_datetime(tasks["Date"], errors="coerce").dt.date

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("⚙️ Settings")

    exam_type = st.radio("Select Exam Type", ["JEE Main (300)", "JEE Advanced (360)"])

    if exam_type == "JEE Main (300)":
        max_per_subject = 100
        total_max = 300
    else:  # JEE Advanced
        max_per_subject = 60  # per subject per paper
        total_max = 360

    candidate_count = st.number_input(
        "Estimated candidates (AIR calc)", 100000, 2_000_000, 1_000_000, step=50_000
    )

    all_students = ["All"] + sorted([s for s in df["Student"].dropna().unique().tolist() if s])
    who = st.selectbox("View for student", all_students, index=0)

    view_df = df if who == "All" else df[df["Student"] == who]

    candidate_count = st.number_input("Estimated candidates (AIR calc)", 100000, 2_000_000, 1_000_000, step=50_000)

    all_students = ["All"] + sorted([s for s in df["Student"].dropna().unique().tolist() if s])
    who = st.selectbox("View for student", all_students, index=0)

    view_df = df if who == "All" else df[df["Student"] == who]

# -------------------- TABS --------------------
tab_dash, tab_add, tab_charts, tab_plan, tab_leader, tab_reports, tab_settings = st.tabs(
    ["Dashboard", "Add Test", "Charts", "Planner", "Leaderboard", "Reports", "Settings"]
)

# -------------------- DASHBOARD --------------------
with tab_dash:
    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)

    latest_total = int(view_df["Total"].iloc[-1]) if not view_df.empty else 0
    perc = estimate_percentile(latest_total, total_max) if latest_total else 0.0
    rank = estimate_rank(perc, candidate_count) if latest_total else 0

    with col1:
        st.metric("Latest Total", f"{latest_total}/{total_max}")
    with col2:
        st.metric("Est. Percentile", f"{perc:.2f}")
    with col3:
        st.metric("Est. AIR", f"{rank if rank else '-'}")
    with col4:
        streak = calc_streak(view_df["Date"].tolist()) if not view_df.empty else 0
        st.metric("Streak (days)", f"{streak}")

    st.divider()

    st.markdown("### 🧠 Weakness Analyzer (last 5 tests)")
    if view_df.empty:
        st.info("Add tests to analyze strengths/weaknesses.")
    else:
        recent = view_df.tail(5)
        means = {
            "Physics": recent["Physics"].mean(),
            "Chemistry": recent["Chemistry"].mean(),
            "Maths": recent["Maths"].mean(),
        }
        weakest = min(means, key=means.get)
        st.write(f"**Weakest (recent): {weakest}**")
        st.write(f"Suggestion: {weakness_tip(weakest)}")

    st.divider()
    st.markdown("### 📅 Calendar Heatmap (Totals)")
    calendar_heatmap(view_df, "Total")

# -------------------- ADD TEST --------------------
with tab_add:
    st.subheader("Add New Test")
    colA, colB, colC, colD, colE, colF = st.columns([1.2,1,1,1,1,1.2])
    with colA:
        student_name = st.text_input("Student", value="You")
    with colB:
        dt = st.date_input("Date", value=date.today())
    with colC:
        phy = st.number_input("Physics", 0, max_per_subject, 0)
    with colD:
        chem = st.number_input("Chemistry", 0, max_per_subject, 0)
    with colE:
        math = st.number_input("Maths", 0, max_per_subject, 0)
    with colF:
        hours = st.number_input("Study Hours (for this test period)", 0.0, 100.0, 0.0, step=0.5)

    if st.button("➕ Save Test"):
        total = int(phy + chem + math)
        new = {
            "Date": dt, "Student": student_name.strip() or "You",
            "Physics": int(phy), "Chemistry": int(chem), "Maths": int(math),
            "Total": total, "StudyHours": float(hours)
        }
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        df = ensure_types(df)
        save_csv(df, DATA_FILE)
        st.success(f"Saved: {student_name} | {dt} | P {phy} | C {chem} | M {math} | Total {total}")

    st.divider()
    st.markdown("#### Current Data")
    st.dataframe(view_df.sort_values("Date"), use_container_width=True)

# -------------------- CHARTS --------------------
with tab_charts:
    st.subheader("Progress Charts")
    if view_df.empty:
        st.info("No data yet.")
    else:
        vdf = view_df.sort_values("Date")
        tests = np.arange(1, len(vdf) + 1)

        st.markdown("**Subject-wise & Total (Line)**")
        fig1, ax1 = plt.subplots()
        ax1.plot(tests, vdf["Physics"], marker="o", label="Physics")
        ax1.plot(tests, vdf["Chemistry"], marker="o", label="Chemistry")
        ax1.plot(tests, vdf["Maths"], marker="o", label="Maths")
        ax1.plot(tests, vdf["Total"], marker="o", linestyle="--", label="Total")
        ax1.set_xlabel("Test #"); ax1.set_ylabel(f"Marks (Max per subject={max_per_subject})")
        ax1.grid(True); ax1.legend()
        st.pyplot(fig1)

        st.markdown("**Per-Test Subject Comparison (Bars)**")
        idx = np.arange(len(vdf)); width = 0.25
        fig2, ax2 = plt.subplots()
        ax2.bar(idx - width, vdf["Physics"], width, label="Physics")
        ax2.bar(idx, vdf["Chemistry"], width, label="Chemistry")
        ax2.bar(idx + width, vdf["Maths"], width, label="Maths")
        ax2.set_xlabel("Test #"); ax2.set_ylabel("Marks")
        ax2.set_xticks(idx, [str(i) for i in tests])
        ax2.legend()
        st.pyplot(fig2)

        st.markdown("**Percentile & AIR Trend**")
        percs = [estimate_percentile(int(t), total_max) for t in vdf["Total"]]
        ranks = [estimate_rank(p, candidate_count) for p in percs]
        fig3 = go.Figure()
        fig3.add_scatter(y=percs, x=list(range(1, len(percs)+1)), mode="lines+markers", name="Percentile")
        fig3.add_scatter(y=ranks, x=list(range(1, len(ranks)+1)), mode="lines+markers",
                         name="AIR (lower is better)", yaxis="y2")
        fig3.update_layout(
            template="plotly_dark" if dark else "plotly",
            yaxis=dict(title="Percentile"),
            yaxis2=dict(title="AIR", overlaying="y", side="right"),
            xaxis=dict(title="Test #"),
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig3, use_container_width=True)

# -------------------- PLANNER --------------------
with tab_plan:
    st.subheader("Weekly Planner & Tasks")
    colp1, colp2 = st.columns([1.5, 1])
    with colp1:
        new_task = st.text_input("Add Task (e.g., Organic Reactions revision)")
        if st.button("Add Task"):
            if new_task.strip():
                row = {"Date": date.today(), "Task": new_task.strip(), "Done": 0}
                tasks = pd.concat([tasks, pd.DataFrame([row])], ignore_index=True)
                save_csv(tasks, TASKS_FILE)
                st.success("Task added!")
    with colp2:
        if not tasks.empty:
            st.write("Mark tasks done by checking boxes below.")

    if tasks.empty:
        st.info("No tasks yet.")
    else:
        for i, r in tasks.sort_values(["Done","Date"]).iterrows():
            checked = st.checkbox(f"{r['Task']}  — ({r['Date']})", value=bool(r["Done"]), key=f"task_{i}")
            tasks.at[i, "Done"] = 1 if checked else 0
        save_csv(tasks, TASKS_FILE)
        done = int(tasks["Done"].sum()); total_tasks = len(tasks)
        st.progress(0 if total_tasks==0 else done/total_tasks)
        st.caption(f"Completed {done}/{total_tasks} tasks")

# -------------------- LEADERBOARD --------------------
with tab_leader:
    st.subheader("Leaderboard (Best total per student)")
    if df.empty:
        st.info("No data.")
    else:
        best = df.sort_values(["Student","Total"], ascending=[True,False]).groupby("Student").head(1)
        board = best[["Student","Total"]].sort_values("Total", ascending=False).reset_index(drop=True)
        st.dataframe(board, use_container_width=True)

# -------------------- REPORTS --------------------
with tab_reports:
    st.subheader("Export / Import / PDF Report")

    colr1, colr2 = st.columns(2)

    with colr1:
        scope = st.radio("Export which data?", ["Filtered (by student)", "All"], horizontal=True)
        exp_df = view_df.copy() if scope.startswith("Filtered") else df.copy()
        csv_bytes = exp_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="jee_prep_data.csv", mime="text/csv")

        if not exp_df.empty:
            pdf_bytes = pdf_report(exp_df)
            st.download_button("⬇️ Download PDF Report", data=pdf_bytes, file_name="JEE_Report.pdf", mime="application/pdf")
        else:
            st.info("No data to include in PDF report yet.")

    with colr2:
        uploaded = st.file_uploader("📥 Import CSV to merge", type=["csv"])
        if uploaded is not None:
            try:
                new_df = pd.read_csv(uploaded)
                needed = {"Date","Student","Physics","Chemistry","Maths","Total"}
                if not needed.issubset(set(new_df.columns)):
                    st.error("CSV missing required columns.")
                else:
                    new_df["Date"] = pd.to_datetime(new_df["Date"], errors="coerce").dt.date
                    if "StudyHours" not in new_df.columns:
                        new_df["StudyHours"] = 0.0
                    st.success("Preview of imported data:")
                    st.dataframe(new_df.head(), use_container_width=True)
                    if st.button("Merge into current data"):
                        merged = pd.concat([df, new_df], ignore_index=True).drop_duplicates()
                        merged = ensure_types(merged)
                        save_csv(merged, DATA_FILE)
                        st.success("✅ Imported & merged! Reload the page to see updated charts.")
            except Exception as e:
                st.error(f"Import failed: {e}")

# -------------------- SETTINGS --------------------
with tab_settings:
    st.subheader("Data Management")
    colS1, colS2 = st.columns(2)
    with colS1:
        if st.button("🧹 Clear ALL data (scores)"):
            df = pd.DataFrame(columns=["Date","Student","Physics","Chemistry","Maths","Total","StudyHours"])
            save_csv(df, DATA_FILE)
            st.warning("All score data cleared. (Charts will be empty until you add tests.)")

    with colS2:
        if who != "All" and not view_df.empty:
            if st.button(f"🧽 Clear data for '{who}' only"):
                df = df[df["Student"] != who]
                save_csv(df, DATA_FILE)
                st.warning(f"All data for '{who}' cleared.")
        else:
            st.info("Select a specific student in the sidebar to clear their data only.")

    st.divider()
    with st.expander("📄 See raw data"):
        st.dataframe(df.sort_values("Date"), use_container_width=True)

    st.caption("Tip: Full dark mode for the app is available in the Streamlit menu → Settings → Theme → Dark.")


        

