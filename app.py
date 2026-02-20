import streamlit as st
import pandas as pd
from pathlib import Path
from io import BytesIO

from fe_core.colors import TEAM_MAP, DRIVER_COLOUR
from fe_core.ingest import read_outing_table_blocks
from fe_core.runs import compute_runs_waits
from fe_core.tyre_sets import label_sets_with_numbers_strict
from fe_core.fastlaps import compute_fastlap_sequences, sequences_to_table
from fe_core.plots import runwait_figure


# --------------------------------------------------------------
# Streamlit page config
# --------------------------------------------------------------
st.set_page_config(
    page_title="FE Engineering Dashboard",
    page_icon="🏁",
    layout="wide"
)

st.title("🏁 Formula E Engineering Dashboard")


# --------------------------------------------------------------
# Sidebar: Upload FP1 / FP2 files
# --------------------------------------------------------------
st.sidebar.header("Upload OutingTables")
fp1_file = st.sidebar.file_uploader("FP1 OutingTable (.xlsx)", type=["xlsx"])
fp2_file = st.sidebar.file_uploader("FP2 OutingTable (.xlsx)", type=["xlsx"])

show_300 = st.sidebar.checkbox("Show 300 kW", True)
show_350 = st.sidebar.checkbox("Show 350 kW", True)


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
from io import BytesIO
import streamlit as st
from fe_core.ingest import read_outing_table_blocks

@st.cache_data(show_spinner=False)
def load_per_driver(uploaded_bytes: bytes):
    """Read an uploaded Excel (bytes) into per-driver blocks; cached by file content."""
    return read_outing_table_blocks(BytesIO(uploaded_bytes))


def render_table_with_ribbons(df: pd.DataFrame, title: str) -> str:
    """HTML table in A+ style with team-coloured left ribbons."""
    rows = []
    rows.append(f"""
    <h3 style="font-family:Segoe UI; color:#001F3F; margin:12px 0 6px 0;">
        {title}
    </h3>
    <table style="font-family:Segoe UI; font-size:15px; border-collapse:collapse; width:100%;">
    <thead style="background:#001F3F; color:white; font-weight:600;">
        <tr>
            <th style="padding:8px 10px; text-align:right;">#</th>
            <th style="padding:8px 10px; text-align:left;">Driver</th>
            <th style="padding:8px 10px; text-align:right;">Best Lap (s)</th>
            <th style="padding:8px 10px; text-align:left;">Sequence</th>
        </tr>
    </thead>
    <tbody>
    """)

    for i, r in df.iterrows():
        band = "#F2F4F7" if i % 2 == 0 else "white"
        rib_color = DRIVER_COLOUR.get(r["Driver"], "#888")

        rows.append(f"""
        <tr style="background:{band}; border-left:6px solid {rib_color};">
            <td style="text-align:right; padding:6px 10px;">{i+1}</td>
            <td style="padding:6px 10px;"><b>{r['Driver']}</b></td>
            <td style="text-align:right; padding:6px 10px;">
                <span style="background:#DFF0D8; padding:2px 6px; border-radius:4px;">
                    <b>{float(r['BestLap_s']):.3f}</b>
                </span>
            </td>
            <td style="padding:6px 10px;">{r['Sequence']}</td>
        </tr>
        """)

    rows.append("</tbody></table>")
    return "\n".join(rows)


# --------------------------------------------------------------
# Tabs
# --------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Session Overview", "Run/Wait + Tyre Sets", "Fast-Lap Sequences"])


# --------------------------------------------------------------
# TAB 1 — Session Overview
# --------------------------------------------------------------
with tab1:
    st.header("Session Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("FP1 Drivers")
        if fp1_file:
            per1 = load_per_driver(fp1_file.getvalue())   
            st.write(sorted(list(per1.keys())))
        else:
            st.info("Upload FP1 file.")

    with col2:
        st.subheader("FP2 Drivers")
        if fp2_file:
            per2 = load_per_driver(fp2_file)
            st.write(sorted(list(per2.keys())))
        else:
            st.info("Upload FP2 file.")


# --------------------------------------------------------------
# TAB 2 — Run/Wait + Tyre Sets
# --------------------------------------------------------------
with tab2:
    st.header("Run/Wait Timeline + Strict Tyre Set Logic")

    session_choice = st.radio("Choose session", ["FP1", "FP2"], horizontal=True)
    file = fp1_file if session_choice == "FP1" else fp2_file

    if not file:
        st.warning(f"Upload {session_choice} file to view.")
    else:
        per_blocks = load_per_driver(file)

        per_struct = {}
        for drv, df in per_blocks.items():
            runs, waits = compute_runs_waits(df)
            set_no, labels = label_sets_with_numbers_strict(runs)
            per_struct[drv] = {
                "runs": runs,
                "waits": waits,
                "run_durs": [(r["end_tod"] - r["start_tod"]) / 60.0 for r in runs],
                "wait_durs": [(w["end_tod"] - w["start_tod"]) / 60.0 for w in waits],
                "tyre_labels": labels,
                "tyre_set_numbers": set_no
            }

        # Team colours (two shades)
        TEAM_COLOURS_2SHADE = {
            'Porsche':  ('#6A0DAD','#A666D6'),
            'Jaguar':   ('#808080','#B0B0B0'),
            'Nissan':   ('#FF66B2','#FF99CC'),
            'Mahindra': ('#D72638','#F15A5A'),
            'DS':       ('#C5A100','#E0C440'),
            'Andretti': ('#66CCFF','#99DDFF'),
            'Citroen':  ('#00AEEF','#80D9FF'),
            'Envision': ('#00A650','#66CDAA'),
            'Kiro':     ('#8B4513','#CD853F'),
            'Lola':     ('#FFD700','#FFE866'),
        }

        fig = runwait_figure(
            per_struct,
            TEAM_MAP,
            TEAM_COLOURS_2SHADE,
            f"{session_choice} — Run/Wait Profile"
        )
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------
# TAB 3 — Fast-Lap Sequences
# --------------------------------------------------------------
with tab3:
    st.header("Fast-Lap Sequences (O/B/P)")

    for sess, file in (("FP1", fp1_file), ("FP2", fp2_file)):
        st.subheader(sess)
        if not file:
            st.info(f"Upload {sess} file.")
            continue

        per_blocks = load_per_driver(file)
        fast_results = compute_fastlap_sequences(per_blocks, powers=(300, 350))

        colA, colB = st.columns(2)

        if show_300:
            df300 = sequences_to_table(fast_results, 300)
            if not df300.empty:
                colA.markdown(render_table_with_ribbons(df300, f"{sess} — 300 kW"), unsafe_allow_html=True)

        if show_350:
            df350 = sequences_to_table(fast_results, 350)
            if not df350.empty:
                colB.markdown(render_table_with_ribbons(df350, f"{sess} — 350 kW"), unsafe_allow_html=True)
