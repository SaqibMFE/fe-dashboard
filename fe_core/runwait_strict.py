import pandas as pd
import plotly.graph_objects as go


# ============================================================
# 1. RUN SEGMENTATION (your original correct logic)
# ============================================================
def compute_runs(df):
    """
    Original engineering-correct run segmentation:
    - Handles OUT, IN, OUT/IN properly
    - Avoids FE-core over-segmentation
    - Produces correct run_durs, wait_durs
    """
    runs = []
    waits = []
    state = "idle"
    curr = None

    for _, row in df.iterrows():
        time_str = str(row["Time"]).strip().upper()
        tod = pd.to_numeric(row["TOD"], errors="coerce")

        if time_str in ("OUT", "OUT/IN"):
            curr = {
                "start_tod": tod,
                "end_tod": None,
                "FL": str(row["FL"]),
                "FR": str(row["FR"]),
                "RL": str(row["RL"]),
                "RR": str(row["RR"]),
            }
            runs.append(curr)
            state = "run"

            if time_str == "OUT/IN":
                curr["end_tod"] = tod
                curr = None
                state = "idle"

        elif time_str == "IN":
            if state == "run" and curr is not None:
                curr["end_tod"] = tod
                curr = None
                state = "idle"

    # Close last open run
    if state == "run" and curr is not None:
        last_tods = pd.to_numeric(df["TOD"], errors="coerce").dropna()
        curr["end_tod"] = last_tods.iloc[-1] if len(last_tods) > 0 else curr["start_tod"]

    # Filter invalid
    runs = [
        r for r in runs
        if pd.notna(r["start_tod"]) and pd.notna(r["end_tod"]) and r["end_tod"] >= r["start_tod"]
    ]

    # Build waits
    waits = []
    for i in range(len(runs) - 1):
        end = runs[i]["end_tod"]
        nxt = runs[i + 1]["start_tod"]
        if pd.notna(end) and pd.notna(nxt) and nxt >= end:
            waits.append({"start_tod": end, "end_tod": nxt})
        else:
            waits.append({"start_tod": end, "end_tod": end})

    return runs, waits



# ============================================================
# 2. STRICT TYRE-SET LOGIC (your original)
# ============================================================
corner_long = {
    "FL": "front left",
    "FR": "front right",
    "RL": "rear left",
    "RR": "rear right",
}

def compute_sets(runs):
    if not runs:
        return [], []

    set_counter = 0
    known_sets = []
    set_numbers = []
    labels = []

    def ids_of(run):
        return {
            "FL": run["FL"],
            "FR": run["FR"],
            "RL": run["RL"],
            "RR": run["RR"],
        }

    def choose_set(cur_map):
        nonlocal set_counter
        cur_ids = set(cur_map.values())
        best = None
        best_overlap = 0

        for meta in known_sets:
            ov = len(cur_ids & meta["idset"])
            if ov > best_overlap:
                best_overlap = ov
                best = meta

        all_known = set().union(*(m["idset"] for m in known_sets)) if known_sets else set()

        if (not known_sets) or len(cur_ids - all_known) == 4:
            set_counter += 1
            meta = {
                "num": set_counter,
                "baseline": cur_map.copy(),
                "idset": set(cur_map.values())
            }
            known_sets.append(meta)
            return meta

        if best is None:
            set_counter += 1
            meta = {
                "num": set_counter,
                "baseline": cur_map.copy(),
                "idset": set(cur_map.values())
            }
            known_sets.append(meta)
            return meta

        return best

    for r in runs:
        cur = ids_of(r)
        chosen = choose_set(cur)
        set_no = chosen["num"]
        base = chosen["baseline"]
        base_ids = set(base.values())
        cur_ids = set(cur.values())

        # As-marked
        if cur == base:
            labels.append(f"Set {set_no} As marked")

        # Sided (same IDs)
        elif cur_ids == base_ids:

            # front sided
            front_swapped = (
                cur["FL"] == base["FR"] and
                cur["FR"] == base["FL"] and
                cur["RL"] == base["RL"] and
                cur["RR"] == base["RR"]
            )

            # rear sided
            rear_swapped = (
                cur["RL"] == base["RR"] and
                cur["RR"] == base["RL"] and
                cur["FL"] == base["FL"] and
                cur["FR"] == base["FR"]
            )

            if front_swapped:
                labels.append(f"Set {set_no} front sided")
            elif rear_swapped:
                labels.append(f"Set {set_no} rear sided")
            else:
                labels.append(f"Set {set_no} sided")

        # Partial / New
        else:
            new_positions = [p for p in ["FL", "FR", "RL", "RR"] if cur[p] not in base_ids]

            if set(new_positions) == {"FL","FR","RL","RR"}:
                labels.append(f"Set {set_no} As marked")

            elif set(new_positions) == {"FL","FR"}:
                labels.append(f"Set {set_no} with new fronts")

            elif set(new_positions) == {"RL","RR"}:
                labels.append(f"Set {set_no} with new rears")

            elif len(new_positions) == 1:
                p = new_positions[0]
                labels.append(f"Set {set_no} with only {corner_long[p]} new")

            else:
                parts = [f"new {corner_long[p]}" for p in new_positions]
                labels.append(f"Set {set_no} with {', '.join(parts)}")

        set_numbers.append(set_no)

    return set_numbers, labels



# ============================================================
# 3. PLOTLY RUN/WAIT PLOT WITH LABELS (correct)
# ============================================================
def plot_runwait(per_struct, team_map, team_colours_2shade, title):
    """
    Plotly stacked bar run/wait profile with correct tyre labels.
    """

    # Driver order: by team pairs
    order = []
    for team, pair in team_map.items():
        order.extend(pair)
    order.extend([d for d in per_struct if d not in order])

    # Colours
    driver_colour = {}
    for team, pair in team_map.items():
        if team in team_colours_2shade:
            shades = team_colours_2shade[team]
            if len(pair) >= 1:
                driver_colour[pair[0]] = shades[0]
            if len(pair) >= 2:
                driver_colour[pair[1]] = shades[1]
    for d in order:
        driver_colour.setdefault(d, "#888888")

    fig = go.Figure()

    for drv in order:
        data = per_struct[drv]
        runs = data["runs"]
        waits = data["waits"]
        run_durs = data["run_durs"]
        wait_durs = data["wait_durs"]
        tyre_labels = data["tyre_labels"]

        base = 0.0
        max_pairs = max(len(runs), len(waits))

        for i in range(max_pairs):

            # RUN
            if i < len(runs):
                dur = run_durs[i]
            
                # ---- Give OUT/IN runs a minimum visual height ----
                if dur == 0:
                    dur = 0.6   
            
                txt = tyre_labels[i] if i < len(tyre_labels) else ""
                fig.add_trace(
                    go.Bar(
                        x=[drv], y=[dur], base=[base],
                        marker=dict(color=driver_colour[drv], line=dict(width=0.3)),
                        text=[txt if dur >= 1.2 else ""],
                        textposition="inside",
                        hovertemplate=f"{drv}<br>{txt}<br>{dur:.1f} min<extra></extra>",
                        showlegend=False
                    )
                )
                base += dur

            # WAIT
            if i < len(waits):
                wdur = wait_durs[i]
                fig.add_trace(
                    go.Bar(
                        x=[drv], y=[wdur], base=[base],
                        marker=dict(color="#D3D3D3", line=dict(width=0.3)),
                        text=[f"{wdur:.1f} min" if wdur >= 0.8 else ""],
                        textposition="inside",
                        hovertemplate=f"{drv}<br>Wait {wdur:.1f} min<extra></extra>",
                        showlegend=False
                    )
                )
                base += wdur

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis=dict(title=None, tickangle=45, categoryorder="array", categoryarray=order),
        yaxis=dict(title="Time (minutes)"),
        margin=dict(t=60, b=120),
        height=600,
    )

    return fig
