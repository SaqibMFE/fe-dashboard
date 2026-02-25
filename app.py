import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
import streamlit.components.v1 as components

# FE Core modules
from fe_core.colors import TEAM_MAP, DRIVER_COLOUR
from fe_core.ingest import read_outing_table_blocks
from fe_core.runs import compute_runs_waits
from fe_core.tyre_sets import label_sets_with_numbers_strict
from fe_core.fastlaps import compute_fastlap_sequences, sequences_to_table
from fe_core.plots import runwait_figure


# ============================================================
#  PLOTLY TYRE-SET PLOT (FP1 + FP2)
# ============================================================
def generate_tyreset_plot_plotly(fp1_bytes: bytes, fp2_bytes: bytes):
    """Generate a Plotly horizontal grouped bar chart
    showing laps per tyre set across FP1 + FP2."""

    # ---- Reader for OutingTable format ----
    def read_outing_table_bytes(data: bytes) -> pd.DataFrame:
        dfw = pd.read_excel(BytesIO(data), engine="openpyxl")
        driver_row_idx = 1
        data_start_idx = 3
        lap_col = dfw.columns[0]

        drivers = []
        for j in range(1, dfw.shape[1]):
            name = dfw.iat[driver_row_idx, j]
            if pd.notna(name):
                drivers.append((str(name).strip(), j))

        records = []
        for drv, j in drivers:
            block_cols = dfw.columns[j:j + 10]
            cols = [lap_col] + list(block_cols)
            block = dfw.loc[data_start_idx:, cols].copy()
            if block.empty:
                continue

            out_cols = ["Lap", "Time", "S1PM", "S2PM", "S3PM",
                        "FL", "FR", "RL", "RR", "Energy", "TOD"]

            block = block.iloc[:, :len(out_cols)]
            block.columns = out_cols
            block["Driver"] = drv
            records.append(block)

        return pd.concat(records, ignore_index=True)

    # ---- Combine FP1 + FP2 ----
    fp1 = read_outing_table_bytes(fp1_bytes)
    fp1["Session"] = "FP1"
    fp2 = read_outing_table_bytes(fp2_bytes)
    fp2["Session"] = "FP2"
    long_df = pd.concat([fp1, fp2], ignore_index=True)

    # ---- Counting rule (Option C) ----
    long_df["Time_str"] = long_df["Time"].astype(str).str.upper().str.strip()
    invalid = {"NAN", "NONE", "", "-"}
    counted = long_df[~long_df["Time_str"].isin(invalid)].copy()

    # ---- Build SetKey (safe) ----
    def set_key(r):
        raw = [r["FL"], r["FR"], r["RL"], r["RR"]]
        tyres = []
        for t in raw:
            if t is None:
                continue
            t = str(t).strip()
            if t == "" or t.lower() == "nan":
                continue
            tyres.append(t)
        if not tyres:
            return "{Unknown}"
        return "{" + ",".join(sorted(tyres)) + "}"

    counted["SetKey"] = counted.apply(set_key, axis=1)
    counted["order_idx"] = counted.groupby("Driver").cumcount()

    oc = (
        counted.groupby(["Driver", "SetKey"])["order_idx"]
        .min()
        .reset_index()
        .sort_values(["Driver", "order_idx"])
    )
    oc["SetNo"] = oc.groupby("Driver").cumcount() + 1
    map_set = {(r.Driver, r.SetKey): r.SetNo for r in oc.itertuples()}
    counted["SetNo"] = counted.apply(lambda r: map_set[(r["Driver"], r["SetKey"])], axis=1)

    # ---- Aggregate ----
    agg = counted.groupby(["Driver", "SetNo"]).size().reset_index(name="Laps")
    totals = agg.groupby("Driver")["Laps"].sum().reset_index(name="Total")
    agg = agg.merge(totals, on="Driver").sort_values(
        ["Total","Driver"], ascending=[False,True]
    )

    # ---- Driver × set matrix ----
    pivot = agg.pivot(index="Driver", columns="SetNo", values="Laps").fillna(0)

    # ============================================================
    #  FE-CORE COLOUR LOGIC — THIS FIXES GREY BARS
    # ============================================================
    driver_colours = {drv: DRIVER_COLOUR.get(drv, "#888888") for drv in pivot.index}

    # ============================================================
    #  Build Plotly figure
    # ============================================================
    fig = go.Figure()
    set_numbers = pivot.columns.tolist()
    drivers = pivot.index.tolist()

    for set_no in set_numbers:
        colours = [driver_colours[drv] for drv in drivers]

        fig.add_trace(
            go.Bar(
                x=pivot[set_no],
                y=drivers,
                name=f"Set {set_no}",
                orientation="h",
                offsetgroup=str(set_no),
                marker=dict(color=colours),
                text=[f"{int(v)} laps" if v > 0 else "" for v in pivot[set_no]],
                textposition="outside"
            )
        )

    fig.update_layout(
        barmode="group",
        title="FP1 + FP2 — Tyre‑Set Laps per Driver (Option C)",
        xaxis_title="Laps",
        yaxis_autorange="reversed",
        height=600,
        margin=dict(t=60, r=40, b=40, l=120)
    )

    return fig


# ============================================================
#  NICE HTML TABLE WITH COLOUR RIBBONS
# ============================================================
def render_table_with_ribbons(df: pd.DataFrame, title: str) -> str:
    """Return a clean HTML table."""
    rows = []
    rows.append(f"""
    <h3 style="font-family:Segoe UI; color:#001F3F; margin:12px 0 6px 0;">
        {title}
    </h3>
    <table style="font-family:Segoe UI; font-size:15px; border-collapse:collapse; width:100%;">
        <thead style="background:#001F3F; color:white; font-weight:600;">
            <tr>
                <th style="text-align:right; padding:6px 10px;">#</th>
                <th style="text-align:left; padding:6px 10px;">Driver</th>
                <th style="text-align:right; padding:6px 10px;">Best Lap (s)</th>
                <th style="text-align:left; padding:6px 10px;">Sequence</th>
            </tr>
        </thead>
        <tbody>
    """)

    for i, r in df.iterrows():
        band = "#F2F4F7" if i % 2 == 0 else "white"
        rib_color = DRIVER_COLOUR.get(str(r["Driver"]), "#888")
        best_str = f"{float(r['BestLap_s']):.3f}" if pd.notna(r["BestLap_s"]) else ""

        rows.append(f"""
            <tr style="background:{band}; border-left:6px solid {rib_color};">
                <td style="text-align:right; padding:6px 10px;">{i+1}</td>
                <td style="padding:6px 10px;"><b>{r['Driver']}</b></td>
                <td style="text-align:right; padding:6px 10px;">
                    <span style="background:#DFF0D8; padding:2px 6px; border-radius:4px;">
                        <b>{best_str}</b>
                    </span>
                </td>
                <td style="padding:6px 10px;">{r['Sequence']}</td>
            </tr>
        """)

    rows.append("</tbody></table>")
    return "\n".join(rows)


# ============================================================
#  STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="FE Engineering Dashboard",
    page_icon="🏁",
    layout="wide"
)
st.title("🏁 Formula E Engineering Dashboard")


# ============================================================
#  SIDEBAR — File Uploads
# ============================================================
st.sidebar.header("Upload OutingTables")
fp1_file = st.sidebar.file_uploader("FP1 OutingTable (.xlsx)", type=["xlsx"])
fp2_file = st.sidebar.file_uploader("FP2 OutingTable (.xlsx)", type=["xlsx"])

show_300 = st.sidebar.checkbox("Show 300 kW", True)
show_350 = st.sidebar.checkbox("Show 350 kW", True)


# ============================================================
#  Cached loader for FE Core
# ============================================================
@st.cache_data(show_spinner=False)
def load_per_driver_from_bytes(uploaded_bytes: bytes):
    return read_outing_table_blocks(BytesIO(uploaded_bytes))


# ============================================================
#  TABS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "Session Overview",
    "Run/Wait + Tyre Sets",
    "Fast-Lap Sequences"
])


# ============================================================
#  TAB 1 — Session Overview
# ============================================================
with tab1:
    st.header("Session Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("FP1 Drivers")
        if fp1_file:
            try:
                per1 = load_per_driver_from_bytes(fp1_file.getvalue())
                st.write(sorted(list(per1.keys())))
            except Exception as e:
                st.error("FP1 load failed.")
                st.exception(e)
        else:
            st.info("Upload FP1 file.")

    with col2:
        st.subheader("FP2 Drivers")
        if fp2_file:
            try:
                per2 = load_per_driver_from_bytes(fp2_file.getvalue())
                st.write(sorted(list(per2.keys())))
            except Exception as e:
                st.error("FP2 load failed.")
                st.exception(e)
        else:
            st.info("Upload FP2 file.")

    st.markdown("---")
        st.subheader("Laptime Standings (300 kW + 350 kW)")
    
        def make_standings_table(per_blocks, title):
            # Compute fastlap structures for this session
            fast_results = compute_fastlap_sequences(per_blocks, powers=(300, 350))
    
            # 300 kW table
            df300 = sequences_to_table(fast_results, 300)
            if not df300.empty:
                st.markdown(f"### {title} — 300 kW")
                df300_show = df300[["Driver", "BestLap_s"]].copy()
                df300_show["BestLap_s"] = df300_show["BestLap_s"].map(
                    lambda x: f"{float(x):.3f}" if pd.notna(x) else ""
                )
                st.table(df300_show)
    
            # 350 kW table
            df350 = sequences_to_table(fast_results, 350)
            if not df350.empty:
                st.markdown(f"### {title} — 350 kW")
                df350_show = df350[["Driver", "BestLap_s"]].copy()
                df350_show["BestLap_s"] = df350_show["BestLap_s"].map(
                    lambda x: f"{float(x):.3f}" if pd.notna(x) else ""
                )
                st.table(df350_show)
    
        # ---- FP1 Standings ----
        if fp1_file:
            try:
                per1 = load_per_driver_from_bytes(fp1_file.getvalue())
                make_standings_table(per1, "FP1")
            except Exception as e:
                st.error("Failed computing FP1 laptime standings.")
                st.exception(e)
    
        # ---- FP2 Standings ----
        if fp2_file:
            try:
                per2 = load_per_driver_from_bytes(fp2_file.getvalue())
                make_standings_table(per2, "FP2")
            except Exception as e:
                st.error("Failed computing FP2 laptime standings.")
                st.exception(e)

# ============================================================
#  TAB 2 — Run/Wait + Tyre Sets
# ============================================================
with tab2:
    st.header("Run/Wait Timeline + Strict Tyre Set Logic")

    session_choice = st.radio("Choose session", ["FP1","FP2"], horizontal=True)
    session_file = fp1_file if session_choice == "FP1" else fp2_file

    if not session_file:
        st.warning(f"Upload {session_choice} file to view run/wait profile.")
    else:
        try:
            per_blocks = load_per_driver_from_bytes(session_file.getvalue())
        except Exception as e:
            st.error("Could not read outing table.")
            st.exception(e)
            per_blocks = None

        if per_blocks:
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

            TEAM_COLOURS_2SHADE = {
                "Porsche": ("#6A0DAD","#A666D6"),
                "Jaguar": ("#808080","#B0B0B0"),
                "Nissan": ("#FF66B2","#FF99CC"),
                "Mahindra": ("#D72638","#F15A5A"),
                "DS": ("#C5A100","#E0C440"),
                "Andretti": ("#66CCFF","#99DDFF"),
                "Citroen": ("#00AEEF","#80D9FF"),
                "Envision": ("#00A650","#66CDAA"),
                "Kiro": ("#8B4513","#CD853F"),
                "Lola": ("#FFD700","#FFE866"),
            }

            fig = runwait_figure(
                per_struct,
                TEAM_MAP,
                TEAM_COLOURS_2SHADE,
                f"{session_choice} — Run/Wait Profile"
            )

            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---- FP1+FP2 Tyre-Set Chart ----
    if fp1_file and fp2_file:
        st.subheader("FP1 + FP2 — Tyre‑Set Laps per Driver")
        try:
            fig_ts = generate_tyreset_plot_plotly(
                fp1_file.getvalue(),
                fp2_file.getvalue()
            )
            st.plotly_chart(fig_ts, use_container_width=True)
        except Exception as e:
            st.error("Could not generate tyre‑set chart.")
            st.exception(e)
    else:
        st.info("Upload both FP1 and FP2 to view tyre‑set usage chart.")


# ============================================================
#  TAB 3 — Fast-Lap Sequences
# ============================================================
with tab3:
    st.header("Fast-Lap Sequences (O/B/P)")

    for sess, file in (("FP1", fp1_file), ("FP2", fp2_file)):
        st.subheader(sess)

        if not file:
            st.info(f"Upload {sess} file.")
            continue

        try:
            per_blocks = load_per_driver_from_bytes(file.getvalue())
        except Exception as e:
            st.error(f"{sess} load failed.")
            st.exception(e)
            continue

        fast_results = compute_fastlap_sequences(per_blocks, powers=(300,350))

        colA, colB = st.columns(2)

        if show_300:
            df300 = sequences_to_table(fast_results, 300)
            if not df300.empty:
                html300 = render_table_with_ribbons(df300, f"{sess} — 300 kW")
                components.html(
                    html300,
                    height=min(120 + 28 * len(df300), 800),
                    scrolling=True
                )

        if show_350:
            df350 = sequences_to_table(fast_results, 350)
            if not df350.empty:
                html350 = render_table_with_ribbons(df350, f"{sess} — 350 kW")
                components.html(
                    html350,
                    height=min(120 + 28 * len(df350), 800),
                    scrolling=True
                )
