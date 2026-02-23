import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
import streamlit.components.v1 as components

from fe_core.colors import TEAM_MAP, DRIVER_COLOUR
from fe_core.ingest import read_outing_table_blocks
from fe_core.runs import compute_runs_waits
from fe_core.tyre_sets import label_sets_with_numbers_strict
from fe_core.fastlaps import compute_fastlap_sequences, sequences_to_table
from fe_core.plots import runwait_figure


# --------------------------------------------------------------
# Plotly FP1+FP2 Tyre-Set Laps Chart
# --------------------------------------------------------------
def generate_tyreset_plot_plotly(fp1_bytes: bytes, fp2_bytes: bytes):
    """
    Returns a Plotly figure for the FP1+FP2 tyre-set laps chart.
    Fully interactive, Streamlit-safe, and does not use matplotlib.
    """

    # --------------------------
    # Internal reader (same logic)
    # --------------------------
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

    # --------------------------
    # Build FP1+FP2 long DF
    # --------------------------
    fp1 = read_outing_table_bytes(fp1_bytes)
    fp1["Session"] = "FP1"
    fp2 = read_outing_table_bytes(fp2_bytes)
    fp2["Session"] = "FP2"
    long_df = pd.concat([fp1, fp2], ignore_index=True)

    # Counting rule
    long_df["Time_str"] = long_df["Time"].astype(str).str.upper().str.strip()
    invalid = {"NAN", "NONE", "", "-"}
    counted = long_df[~long_df["Time_str"].isin(invalid)].copy()

    # ============= SAFE SetKey builder =============
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

    agg = counted.groupby(["Driver", "SetNo"]).size().reset_index(name="Laps")
    totals = agg.groupby("Driver")["Laps"].sum().reset_index(name="Total")
    agg = agg.merge(totals, on="Driver").sort_values(["Total", "Driver"], ascending=[False, True])

    pivot = agg.pivot(index="Driver", columns="SetNo", values="Laps").fillna(0)

    # --------------------------
    # Build PLOTLY FIGURE
    # --------------------------
    fig = go.Figure()
    set_numbers = pivot.columns.tolist()
    drivers = pivot.index.tolist()

    for set_no in set_numbers:
        fig.add_trace(
            go.Bar(
                x=pivot[set_no],
                y=drivers,
                name=f"Set {set_no}",
                orientation="h",
                offsetgroup=str(set_no),
                text=[
                    f"Set {set_no} — {int(v)} laps" if v > 0 else ""
                    for v in pivot[set_no]
                ],
                textposition="outside",
            )
        )

    fig.update_layout(
        barmode="group",
        title="FP1 + FP2 — Tyre‑Set Laps per Driver (Option C)",
        xaxis_title="Laps",
        yaxis_autorange="reversed",
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        margin=dict(t=60, r=20, b=40, l=80),
    )

    return fig


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
# Cached loader
# --------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_per_driver_from_bytes(uploaded_bytes: bytes):
    return read_outing_table_blocks(BytesIO(uploaded_bytes))


# --------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------
st.sidebar.header("Upload OutingTables")
fp1_file = st.sidebar.file_uploader("FP1 OutingTable (.xlsx)", type=["xlsx"])
fp2_file = st.sidebar.file_uploader("FP2 OutingTable (.xlsx)", type=["xlsx"])

show_300 = st.sidebar.checkbox("Show 300 kW", True)
show_350 = st.sidebar.checkbox("Show 350 kW", True)


# --------------------------------------------------------------
# Helper for custom HTML tables
# --------------------------------------------------------------
def render_table_with_ribbons(df: pd.DataFrame, title: str) -> str:

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
        rib_color = DRIVER_COLOUR.get(str(r["Driver"]), "#888")
        best_str = f"{float(r['BestLap_s']):.3f}" if pd.notna(r["BestLap_s"]) else ""

        rows.append(f"""
            <tr style="background:{band}; border-left:6px solid {rib_color};">
                <td style="text-align:right; padding:6px 10px;">{i+1}</td>
                <td style="padding:6px 10px;"><b>{r['Driver']}</b></td>
