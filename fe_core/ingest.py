import pandas as pd
from pathlib import Path

HEADER_TOKENS=['Time','S1 PM','S2 PM','S3 PM','FL','FR','RL','RR','Energy','TOD']

def read_outing_table_blocks(source):
    raw = pd.read_excel(source, header=None, engine='openpyxl')
    col0 = raw.iloc[:,0].astype(str).str.strip().str.upper()

    idx_driver = int(raw.index[col0.eq('DRIVER')][0])
    idx_lap    = int(raw.index[col0.eq('LAP')][0])

    row_driver = raw.iloc[idx_driver].tolist()
    headers    = raw.iloc[idx_lap].astype(str).str.strip().str.upper().tolist()

    start_cols = [j for j,v in enumerate(headers) if v=='TIME']
    start_data = idx_lap+1

    per_driver={}
    for sc in start_cols:
        code=str(row_driver[sc]).strip()
        if code in ('','nan','NaN','LAP','Driver'): continue
        block=raw.iloc[start_data:, sc:sc+10].copy()
        if block.shape[1]<10: continue
        block.columns=HEADER_TOKENS
        block.insert(0,'Lap', raw.iloc[start_data:,0].values)
        per_driver[code]=block

    return per_driver
