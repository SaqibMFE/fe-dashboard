import pandas as pd

def compute_runs_waits(df):
    runs=[]; waits=[]; state='idle'; curr=None

    for _, row in df.iterrows():
        time_str = str(row['Time']).strip().upper()
        tod = pd.to_numeric(row['TOD'], errors='coerce')

        if time_str in ('OUT','OUT/IN'):
            curr={'start_tod':tod,'end_tod':None,
                  'FL':str(row['FL']),'FR':str(row['FR']),
                  'RL':str(row['RL']),'RR':str(row['RR'])}
            runs.append(curr)
            state='run'
            if time_str=='OUT/IN':
                curr['end_tod']=tod
                curr=None
                state='idle'

        elif time_str=='IN':
            if state=='run' and curr is not None:
                curr['end_tod']=tod
                curr=None
                state='idle'

    if state=='run' and curr is not None:
        last=pd.to_numeric(df['TOD'], errors='coerce').dropna()
        curr['end_tod']=last.iloc[-1] if len(last)>0 else curr['start_tod']

    runs=[r for r in runs if pd.notna(r['start_tod']) and pd.notna(r['end_tod']) and r['end_tod']>=r['start_tod']]

    for i in range(len(runs)-1):
        end=runs[i]['end_tod']; nxt=runs[i+1]['start_tod']
        waits.append({'start_tod':end,'end_tod':nxt})

    return runs, waits
