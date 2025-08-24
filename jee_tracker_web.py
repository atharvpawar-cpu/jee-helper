import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_FILE = "scores.csv"

# -------------------- DATA HANDLING --------------------
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame(columns=["Student", "Exam", "Physics", "Chemistry", "Maths", "Total"])

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

df = load_data()

# -------------------- SIDEBAR SETTINGS --------------------
with st.sidebar:
    st.header("⚙️ Settings")

    exam_type = st.selectbox("Select Exam Type", ["JEE Main (300)", "JEE Advanced"])
    candidate_count = st.number_input("Estimated candidates (AIR calc)", 100000, 2_000_000, 1_000_000, step=50_000)

    # set max marks depending on exam
    if exam_type == "JEE Main (300)":
        max_per_subject = 100
        total_max = 300
    else:  # JEE Advanced
        max_per_subject = 120  # 60 × 2 papers
        total_max = 360

    all_students = ["All"] + sorted([s for s in df["Student"].dropna().unique().tolist() if s])
    who = st.selectbox("View for student", all_students, index=0)
    view_df = df if who == "All" else df[df["Student"] == who]
    # ---- Exam-specific scoring caps ----
if exam_type == "JEE Main (300)":
    max_per_subject = 100   # per subject
    total_max = 300
else:  # JEE Advanced
    max_per_subject = 120   # 60 per paper x 2 papers
    total_max = 360


# -------------------- ADD SCORE FORM --------------------
with st.form("add_score"):
    st.subheader("➕ Add New Score")

    name = st.text_input("Student Name")

    if exam_type == "JEE Main (300)":
        physics = st.number_input("Physics (out of 100)", 0, 100, 0)
        chemistry = st.number_input("Chemistry (out of 100)", 0, 100, 0)
        maths = st.number_input("Maths (out of 100)", 0, 100, 0)
        total = physics + chemistry + maths

    else:  # JEE Advanced
        st.markdown("**Paper 1**")
        phy1 = st.number_input("Physics (P1, out of 60)", 0, 60, 0)
        chem1 = st.number_input("Chemistry (P1, out of 60)", 0, 60, 0)
        math1 = st.number_input("Maths (P1, out of 60)", 0, 60, 0)

        st.markdown("**Paper 2**")
        phy2 = st.number_input("Physics (P2, out of 60)", 0, 60, 0)
        chem2 = st.number_input("Chemistry (P2, out of 60)", 0, 60, 0)
        math2 = st.number_input("Maths (P2, out of 60)", 0, 60, 0)

        physics = phy1 + phy2
        chemistry = chem1 + chem2
        maths = math1 + math2
        total = physics + chemistry + maths

    submitted = st.form_submit_button("Save Score")

    if submitted and name:
        new_row = pd.DataFrame([{
            "Student": name,
            "Exam": exam_type,
            "Physics": physics,
            "Chemistry": chemistry,
            "Maths": maths,
            "Total": total
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        save_data(df)
        st.success(f"✅ Saved {name}'s score: {total}/{total_max}")

# -------------------- TABS --------------------
tab_dash, tab_add, tab_charts, tab_plan, tab_leader, tab_reports, tab_settings = st.tabs(
    ["Dashboard", "Add Test", "Charts", "Planner", "Leaderboard", "Reports", "Settings"]
)

with tab_dash:
    st.header("📊 Dashboard")
    if not view_df.empty:
        st.dataframe(view_df)
    else:
        st.info("No scores available yet.")

with tab_add:
    st.header("➕ Add Score (use form above)")

with tab_charts:
    st.header("📈 Charts")
    if not view_df.empty:
        fig, ax = plt.subplots()
        view_df.groupby("Student")["Total"].plot(legend=True)
        ax.set_ylabel(f"Marks (Max {total_max})")
        st.pyplot(fig)
    else:
        st.info("No data to plot.")

with tab_plan:
    st.header("📝 Study Planner (coming soon)")

with tab_leader:
    st.header("🏆 Leaderboard")
    if not df.empty:
        leaderboard = df.groupby("Student")["Total"].max().sort_values(ascending=False).reset_index()
        st.table(leaderboard)
    else:
        st.info("No scores yet.")

with tab_reports:
    st.header("📑 Reports (coming soon)")

with tab_settings:
    st.header("⚙️ Settings are in the sidebar")


