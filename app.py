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

import plotly.graph_objects as go

def runwait_figure_with_labels(
    per_struct: dict,
    team_map: dict,
    team_colours_2shade: dict,
    title: str,
    run_label_threshold_min: float = 1.2,
    wait_label_threshold_min: float = 0.6,
):
    """
    Build a vertically-stacked run/wait timeline with labels:
      - RUN: tyre set label
      - WAIT: 'X.X min'
    Labels are shown inside the bar if segment is long enough; always present on hover.
    """

    # ---- Driver plotting order: by team pairs, then any extras ----
    order = []
    for team, pair in team_map.items():
        order.extend(pair)
    # add any drivers not listed in team_map (if any)
    order.extend([d for d in per_struct.keys() if d not in order])

    # ---- Resolve per-driver colour (light/dark by position in team pair) ----
    driver_colour = {}
    for team, pair in team_map.items():
        if team not in team_colours_2shade:
            continue
        shades = team_colours_2shade[team]  # (light, dark)
        if len(pair) >= 1:
            driver_colour[pair[0]] = shades[0]
        if len(pair) >= 2:
            driver_colour[pair[1]] = shades[1]
    # Fallback grey
    for d in order:
        driver_colour.setdefault(d, "#888888")

    fig = go.Figure()

    # ---- For each driver: stack bars by progressively increasing base ----
    for drv in order:
        data = per_struct.get(drv)
        if not data:
            continue

        runs = data.get("run_durs", [])
        waits = data.get("wait_durs", [])
        labels = data.get("tyre_labels", [])

        base = 0.0
        # Interleave RUN i then WAIT i (if present)
        max_pairs = max(len(runs), len(waits))
        for i in range(max_pairs):
            # RUN i
            if i < len(runs):
                dur = runs[i] or 0.0
                txt = labels[i] if i < len(labels) else ""
                show_text = (dur >= run_label_threshold_min)
                fig.add_trace(
                    go.Bar(
                        x=[drv],
                        y=[dur],
                        base=[base],
                        marker=dict(color=driver_colour[drv], line=dict(width=0.3, color="black")),
                        name=f"{drv} run {i+1}",
                        text=[txt if show_text else ""],
                        textposition="inside",
                        textfont=dict(size=10),
                        hovertemplate=(
                            f"<b>{drv}</b><br>"
                            f"Run {i+1}: {{y:.1f}} min<br>"
                            f"{txt}<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )
                base += dur

            # WAIT i
            if i < len(waits):
                wdur = waits[i] or 0.0
                wait_txt = f"{wdur:.1f} min"
                show_text = (wdur >= wait_label_threshold_min)
                fig.add_trace(
                    go.Bar(
                        x=[drv],
                        y=[wdur],
                        base=[base],
                        marker=dict(color="#D3D3D3", line=dict(width=0.3, color="black")),
                        name=f"{drv} wait {i+1}",
                        text=[wait_txt if show_text else ""],
                        textposition="inside",
                        textfont=dict(size=10),
                        hovertemplate=(
                            f"<b>{drv}</b><br>"
                            f"Wait {i+1}: {{y:.1f}} min<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )
                base += wdur

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis=dict(
            title=None,
            tickangle=45,
            categoryorder="array",
            categoryarray=order
        ),
        yaxis=dict(
            title="Time (minutes)",
            gridcolor="#e0e0e0",
            zeroline=True
        ),
        margin=dict(t=60, r=20, b=80, l=60),
        height=600
    )

    return fig

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
    """HTML table in FE style with team-coloured ribbons + FastLap_RunNumber support."""

    has_fastlap_col = "FastLap_RunNumber" in df.columns

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
    """)

    if has_fastlap_col:
        rows.append("""
                <th style="padding:8px 10px; text-align:center;">FastLap Run #</th>
        """)

    rows.append("""
            </tr>
        </thead>
        <tbody>
    """)

    for i, r in df.iterrows():
        band = "#F2F4F7" if i % 2 == 0 else "white"
        rib_color = DRIVER_COLOUR.get(str(r["Driver"]), "#888")
        best_str = f"{float(r['BestLap_s']):.3f}" if pd.notna(r["BestLap_s"]) else ""
        seq_html = r["Sequence"]

        rows.append(f"""
            <tr style="background:{band}; border-left:6px solid {rib_color};">
                <td style="text-align:right; padding:6px 10px;">{i+1}</td>
                <td style="padding:6px 10px;"><b>{r['Driver']}</b></td>
                <td style="text-align:right; padding:6px 10px;">
                    <span style="background:#DFF0D8; padding:2px 6px; border-radius:4px;">
                        <b>{best_str}</b>
                    </span>
                </td>
                <td style="padding:6px 10px;">{seq_html}</td>
        """)

        if has_fastlap_col:
            rows.append(
                f'<td style="text-align:center; padding:6px 10px;">{r["FastLap_RunNumber"]}</td>'
            )

        rows.append("</tr>")

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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Session Overview",
    "Run/Wait + Tyre Sets",
    "Fast-Lap Sequences",
    "Race",
    "Qualifying"
])


# ============================================================
#  TAB 1 — Session Overview
# ============================================================

with tab1:
    st.header("Session Overview")

    # ----------------------------
    # Session selection (FP1 / FP2)
    # ----------------------------
    session_choice = st.radio(
    "Choose session",
    ["FP1", "FP2"],
    horizontal=True,
    key="tab1_session"          # <-- unique key for Tab 1
)

    # Determine which file to use
    file = fp1_file if session_choice == "FP1" else fp2_file

    if not file:
        st.info(f"Upload {session_choice} file to view standings.")
    else:
        try:
            per_blocks = load_per_driver_from_bytes(file.getvalue())
        except Exception as e:
            st.error(f"{session_choice} load failed.")
            st.exception(e)
            per_blocks = None

        if per_blocks:

            # ----------------------------------
            # Get fastlap sequence results
            # ----------------------------------
            fast_results = compute_fastlap_sequences(per_blocks, powers=(300, 350))

            df300 = sequences_to_table(fast_results, 300)
            df350 = sequences_to_table(fast_results, 350)

            # Format numeric values
            if not df300.empty:
                df300_show = df300[["Driver", "BestLap_s"]].copy()
                df300_show["BestLap_s"] = df300_show["BestLap_s"].map(
                    lambda x: f"{float(x):.3f}" if pd.notna(x) else ""
                )
            else:
                df300_show = pd.DataFrame(columns=["Driver", "BestLap_s"])

            if not df350.empty:
                df350_show = df350[["Driver", "BestLap_s"]].copy()
                df350_show["BestLap_s"] = df350_show["BestLap_s"].map(
                    lambda x: f"{float(x):.3f}" if pd.notna(x) else ""
                )
            else:
                df350_show = pd.DataFrame(columns=["Driver", "BestLap_s"])

            # ----------------------------------
            # Display side-by-side standings
            # ----------------------------------
            st.subheader(f"{session_choice} — Laptime Standings")

            colA, colB = st.columns(2)

            with colA:
                st.markdown("### 300 kW")
                st.table(df300_show)

            with colB:
                st.markdown("### 350 kW")
                st.table(df350_show)


# ============================================================
#  TAB 2 — Run/Wait + Tyre Sets
# ============================================================
with tab2:
    st.header("Run/Wait Timeline + Strict Tyre Set Logic")

    session_choice = st.radio(
        "Choose session",
        ["FP1", "FP2"],
        horizontal=True,
        key="tab2_session"
    )
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

            # ---------------------------------------------------------
            # USE YOUR NEW FE-CORE STRICT MODULE
            # ---------------------------------------------------------
            from fe_core.runwait_strict import compute_runs, compute_sets, plot_runwait

            per_struct = {}
            for drv, df in per_blocks.items():

                # --- Use your correct engineering segmentation logic ---
                runs, waits = compute_runs(df)

                # --- Strict tyre-set logic ---
                set_no, labels = compute_sets(runs)

                per_struct[drv] = {
                    "runs": runs,
                    "waits": waits,
                    "run_durs": [(r["end_tod"] - r["start_tod"]) / 60.0 for r in runs],
                    "wait_durs": [(w["end_tod"] - w["start_tod"]) / 60.0 for w in waits],
                    "tyre_labels": labels,
                    "tyre_set_numbers": set_no
                }

            # --------------------------------------------------------------------
            # Plot with new strict run/wait function (correct labels, correct sets)
            # --------------------------------------------------------------------

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
            
            fig = plot_runwait(
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

    # ------------------------------
    # Session selector for this tab
    # ------------------------------
    session_choice = st.radio(
        "Choose session",
        ["FP1", "FP2"],
        horizontal=True,
        key="tab3_session"
    )

    session_file = fp1_file if session_choice == "FP1" else fp2_file

    if not session_file:
        st.info(f"Upload {session_choice} file to view fast-lap sequences.")
    else:
        try:
            per_blocks = load_per_driver_from_bytes(session_file.getvalue())
        except Exception as e:
            st.error(f"{session_choice} load failed.")
            st.exception(e)
            per_blocks = None

        if per_blocks:
            fast_results = compute_fastlap_sequences(per_blocks, powers=(300,350))

            from fe_core.runwait_strict import compute_runs  # correct run splitter

            # ----------------------------------------------------------------------
            # Enhance FastLap table: Bold fastest P + True RunNumber from session
            # ----------------------------------------------------------------------
            def enhance_fastlap_table(df, power_kW):
                if df.empty:
                    return df

                df = df.copy()
                bold_sequences = []
                run_numbers = []

                for idx, row in df.iterrows():
                    drv = row["Driver"]
                    best = fast_results[drv][power_kW]["best"]
                    seq_string = fast_results[drv][power_kW]["sequence"]
                    tokens = seq_string.split()

                    # Find fastest P index (last P in sequence)
                    fastest_index = None
                    for i, tok in enumerate(tokens):
                        if tok == "P":
                            fastest_index = i

                    # -------------------------------------
                    # Compute real run number from session
                    # -------------------------------------
                    df_driver = per_blocks[drv]  # outing table
                    df_driver2 = df_driver.copy()
                    df_driver2["Time_val"] = pd.to_numeric(df_driver2["Time"], errors="coerce")
                    df_driver2["Power"] = pd.to_numeric(df_driver2["S1 PM"], errors="coerce")

                    mask = (df_driver2["Power"] == power_kW) & (df_driver2["Time_val"] == best)
                    try:
                        best_idx = df_driver2[mask].index[0]
                    except:
                        run_numbers.append("")
                        bold_sequences.append(seq_string)
                        continue

                    runs, _ = compute_runs(df_driver2)

                    real_run_no = ""
                    best_tod = df_driver2.loc[best_idx, "TOD"]

                    for r_i, r in enumerate(runs, start=1):
                        if r["start_tod"] <= best_tod <= r["end_tod"]:
                            real_run_no = r_i
                            break

                    run_numbers.append(real_run_no)

                    # Bold only the fastest P
                    bold_tokens = []
                    for i, tok in enumerate(tokens):
                        if tok == "P" and i == fastest_index:
                            bold_tokens.append("<b>P</b>")
                        else:
                            bold_tokens.append(tok)
                    bold_sequences.append(" ".join(bold_tokens))

                df["Sequence"] = bold_sequences
                df["FastLap_RunNumber"] = run_numbers

                return df

            # ----------------------------------------------------------------------
            # Render 300 & 350 tables side by side
            # ----------------------------------------------------------------------
            colA, colB = st.columns(2)

            # -------- 300 kW --------
            if show_300:
                df300 = sequences_to_table(fast_results, 300)
                df300 = enhance_fastlap_table(df300, 300)
                if not df300.empty:
                    html300 = render_table_with_ribbons(df300, f"{session_choice} — 300 kW")
                    components.html(
                        html300,
                        height=min(120 + 28 * len(df300), 800),
                        scrolling=True
                    )

            # -------- 350 kW --------
            if show_350:
                df350 = sequences_to_table(fast_results, 350)
                df350 = enhance_fastlap_table(df350, 350)
                if not df350.empty:
                    html350 = render_table_with_ribbons(df350, f"{session_choice} — 350 kW")
                    components.html(
                        html350,
                        height=min(120 + 28 * len(df350), 800),
                        scrolling=True
                    )


# ============================================================
# TAB 4 — FULL RACE POSITION MATRIX (P × Lap)
# ============================================================
with tab4:
    st.header("Race Position Matrix (Pos × Lap)")

    race_file = st.file_uploader(
        "Upload Race Lap Chart (.xlsx)", 
        type=["xlsx"], 
        key="race_file_matrix"
    )

    if not race_file:
        st.info("Upload the Race Lap Chart file to view the position matrix.")
    else:
        import pandas as pd

        # Load the race file
        try:
            df = pd.read_excel(race_file, engine="openpyxl", header=None)
        except Exception as e:
            st.error("Could not read race file.")
            st.exception(e)
            st.stop()

        # Lap columns begin at index 2
        lap_cols = df.columns[2:]
        num_laps = len(lap_cols)

        # ----------------------------------------------------------
        # Helper: detect the unwanted numeric row (1,2,3,...)
        # ----------------------------------------------------------
        def is_lap_header_row(values):
            """Detects rows like [1,2,3,...] which must be skipped."""
            try:
                numeric = pd.to_numeric(pd.Series(values), errors="coerce")
            except:
                return False
            if numeric.isna().any():
                return False
            ints = numeric.astype(int).tolist()
            return ints == list(range(1, len(ints) + 1))

        # ----------------------------------------------------------
        # Build table, skipping:
        #   • fully empty rows
        #   • the lap-number header row
        # ----------------------------------------------------------
        table = []
        for r in range(len(df)):
            row_vals = df.iloc[r, 2:].tolist()

            if all(pd.isna(x) for x in row_vals):
                continue

            if is_lap_header_row(row_vals):
                continue

            table.append(row_vals)

        # ----------------------------------------------------------
        # FE Team colour mapping
        # ----------------------------------------------------------
        driver_to_colour = {}
        for team, pair in TEAM_MAP.items():
            shades = TEAM_COLOURS_2SHADE.get(team)
            if not shades:
                continue
            if len(pair) >= 1:
                driver_to_colour[pair[0]] = shades[0]
            if len(pair) >= 2:
                driver_to_colour[pair[1]] = shades[1]

        # fallback
        for row in table:
            for drv in row:
                if pd.isna(drv):
                    continue
                if drv not in driver_to_colour:
                    driver_to_colour[drv] = "#CCCCCC"

        # ----------------------------------------------------------
        # HTML table build
        # ----------------------------------------------------------
        html = """
        <table style="border-collapse:collapse; font-family:Segoe UI; font-size:12px;">
            <thead>
                <tr>
                    <th style='padding:4px; border:1px solid #222;'>P</th>
        """

        # Lap header row (Lap 1, Lap 2 …)
        for lap in range(1, num_laps + 1):
            html += f"<th style='padding:4px; border:1px solid #222;'>Lap {lap}</th>"

        html += "</tr></thead><tbody>"

        # Position rows
        for pos_idx, row in enumerate(table, start=1):
            html += (
                f"<tr>"
                f"<td style='padding:4px; border:1px solid #222; font-weight:bold;'>{pos_idx}</td>"
            )

            for drv in row:
                if pd.isna(drv):
                    html += "<td style='padding:4px; border:1px solid #555; background:white;'></td>"
                else:
                    color = driver_to_colour.get(str(drv), "#ccc")
                    html += (
                        f"<td style='padding:4px; border:1px solid #555; "
                        f"background:{color}; text-align:center; font-weight:bold;'>"
                        f"{drv}</td>"
                    )
            html += "</tr>"

        html += "</tbody></table>"

        st.markdown(html, unsafe_allow_html=True)

# ============================================================
# TAB 5 — QUALIFYING (300 kW ONLY) — Full run sequence with fastest lap in bold
# ============================================================
with tab5:
    st.header("Qualifying — 300 kW (Full Run Sequence, Fastest Lap in Bold)")

    qual_file = st.file_uploader(
        "Upload Qualifying OutingTable (.xlsx)",
        type=["xlsx"],
        key="qualifying_file"
    )

    if not qual_file:
        st.info("Upload the Qualifying OutingTable file.")
    else:
        try:
            # Load per-driver blocks (same loader you already use)
            per_blocks = load_per_driver_from_bytes(qual_file.getvalue())
        except Exception as e:
            st.error("Could not read qualifying file.")
            st.exception(e)
            per_blocks = None

        if per_blocks:
            from fe_core.runwait_strict import compute_runs         # strict run splitter
            from fe_core.fastlaps import compute_fastlap_sequences, sequences_to_table

            # -----------------------------------------------------------
            # 1) Use the SAME sequence engine as in "Fast-Lap Sequences"
            #    (compute ONLY for 300 kW)
            # -----------------------------------------------------------
            fast_results = compute_fastlap_sequences(per_blocks, powers=(300,))
            df300 = sequences_to_table(fast_results, 300)
            if df300.empty:
                st.info("No valid 300 kW laps found in the qualifying file.")
                st.stop()

            rows = []

            # -----------------------------------------------------------
            # 2) For each driver:
            #    • identify the run that contains the fastest 300 kW lap
            #    • take ONLY that run's sequence (O/B/P) and bold the fastest lap
            # -----------------------------------------------------------
            for _, row in df300.iterrows():
                drv = row["Driver"]
                best = row["BestLap_s"]  # numeric (seconds)

                # Full 300 kW sequence for the session from the SAME logic as Tab 3
                seq_string = fast_results[drv][300]["sequence"] if (drv in fast_results and 300 in fast_results[drv]) else ""
                tokens_all = seq_string.split() if isinstance(seq_string, str) else []

                # Build chronologically ordered list of (tod, time_val) for 300 kW laps
                df_driver = per_blocks[drv].copy()
                df_driver["Time_val"] = pd.to_numeric(df_driver["Time"], errors="coerce")
                df_driver["Power"]    = pd.to_numeric(df_driver["S1 PM"], errors="coerce")

                laps300 = df_driver[(df_driver["Power"] == 300) & (df_driver["Time_val"].notna())].copy()
                laps300 = laps300.sort_values("TOD")  # ensure chronological

                tod_list   = laps300["TOD"].tolist()
                time_list  = laps300["Time_val"].tolist()

                # Align tokens with laps; fallback to full-session sequence if mismatch
                use_fallback_full = (len(tokens_all) != len(tod_list))

                # Find best lap TOD (for the 300 kW best time)
                best_tod = None
                try:
                    # If multiple equal times exist, take the first occurrence in time order
                    best_idx = laps300.index[laps300["Time_val"] == best][0]
                    best_tod = df_driver.loc[best_idx, "TOD"]
                except Exception:
                    # If we can't locate the exact row, we will still render using fallback
                    use_fallback_full = True

                # Strict run segmentation — to get the run window + run number
                runs, _ = compute_runs(df_driver)
                run_window = None
                run_no = ""

                if best_tod is not None:
                    for r_i, r in enumerate(runs, start=1):
                        if r["start_tod"] <= best_tod <= r["end_tod"]:
                            run_window = r
                            run_no = r_i
                            break

                # -------------------------------------------------------
                # Build the run-level sequence using the same O/B/P tokens:
                #   - If alignment is perfect: slice tokens by TOD window
                #   - Otherwise, fallback to full-session sequence
                # -------------------------------------------------------
                seq_out = ""
                if (not use_fallback_full) and run_window is not None:
                    # Map tokens to TOD via zipping (same chronological order)
                    pairs = list(zip(tod_list, tokens_all))  # [(tod, token), ...]
                    run_pairs = [(tod, tok) for (tod, tok) in pairs
                                 if (run_window["start_tod"] <= tod <= run_window["end_tod"])]

                    if run_pairs:
                        seq_tokens = []
                        for tod, tok in run_pairs:
                            # Bold the token for the actual fastest 300 kW lap
                            if best_tod is not None and tod == best_tod:
                                # If the logic labeled it 'P', bold P; otherwise bold the token we have
                                seq_tokens.append("<b>P</b>" if tok == "P" else f"<b>{tok}</b>")
                            else:
                                seq_tokens.append(tok)
                        seq_out = " ".join(seq_tokens)
                    else:
                        use_fallback_full = True

                if use_fallback_full:
                    # Fallback: show full sequence and bold the last 'P' (same as Tab 3 convention)
                    toks = tokens_all[:]
                    if toks:
                        try:
                            fastest_index = max(i for i, t in enumerate(toks) if t == "P")
                            toks[fastest_index] = "<b>P</b>"
                        except ValueError:
                            # No P in sequence — bold nothing special
                            pass
                    seq_out = " ".join(toks)

                rows.append({
                    "Driver": drv,
                    "BestLap_s": best,
                    "Sequence": seq_out,               # full run sequence (fastest in bold)
                    "FastLap_RunNumber": run_no        # optional, displayed if present
                })

            dfQ = pd.DataFrame(rows).sort_values("BestLap_s").reset_index(drop=True)

            # -----------------------------------------------------------
            # 3) Render with your existing styled HTML table helper
            # -----------------------------------------------------------
            htmlQ = render_table_with_ribbons(
                dfQ,
                "Qualifying — 300 kW (Run Sequence; Fastest Lap in Bold)"
            )
            # Height scales with rows but capped for usability
            components.html(htmlQ, height=min(120 + 28 * len(dfQ), 900), scrolling=True)
``
