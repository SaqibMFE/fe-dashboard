import pandas as pd

def compute_fastlap_sequences(per_driver_blocks, powers=(300,350)):
    results = {}

    for drv, df in per_driver_blocks.items():
        df=df.copy()
        df['Time_str']=df['Time'].astype(str).str.upper().str.strip()
        df['Time_val']=pd.to_numeric(df['Time'], errors='coerce')
        df['Power']=pd.to_numeric(df['S1 PM'], errors='coerce')

        sub300=df[(df['Power']==300) & df['Time_val'].notna()]
        if sub300.empty:
            results[drv]={}
            continue

        best300 = float(sub300['Time_val'].min())
        thresh  = best300 * 1.025

        drv_res={}
        for p in powers:
            subp=df[(df['Power']==p) & df['Time_val'].notna()]
            if subp.empty: continue

            best=float(subp['Time_val'].min())
            idx = int(subp['Time_val'].idxmin())

            i=idx
            while i>=0 and df.loc[i,'Time_str']!='OUT':
                i-=1
            start=i

            seq=[]
            for j in range(start, idx+1):
                tstr=df.loc[j,'Time_str']
                if tstr=='OUT':
                    seq.append('O')
                else:
                    t=df.loc[j,'Time_val']
                    seq.append('P' if (pd.notna(t) and t<=thresh) else 'B')

            drv_res[p]={'best':best,'sequence':' '.join(seq)}

        results[drv]=drv_res
    return results


def sequences_to_table(results, power):
    rows=[]
    for drv, d in results.items():
        if power not in d: continue
        rows.append({'Driver':drv,'BestLap_s':d[power]['best'],'Sequence':d[power]['sequence']})
    if not rows:
        return pd.DataFrame(columns=['Driver','BestLap_s','Sequence'])
    return pd.DataFrame(rows).sort_values('BestLap_s').reset_index(drop=True)
