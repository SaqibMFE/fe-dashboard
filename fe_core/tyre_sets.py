import pandas as pd

CORNER_LONG={'FL':'front left','FR':'front right','RL':'rear left','RR':'rear right'}

def label_sets_with_numbers_strict(runs):
    if not runs: return [], []

    set_counter=0
    known_sets=[]
    set_numbers=[]; labels=[]

    def ids_of(r):
        return {'FL':r['FL'],'FR':r['FR'],'RL':r['RL'],'RR':r['RR']}

    def choose_set(cur):
        nonlocal set_counter, known_sets
        cur_ids=set(cur.values())
        best=None; best_overlap=0

        for meta in known_sets:
            ov=len(cur_ids & meta['idset'])
            if ov>best_overlap:
                best_overlap=ov; best=meta

        all_known=set().union(*[m['idset'] for m in known_sets]) if known_sets else set()
        if (not known_sets) or len(cur_ids-all_known)==4:
            set_counter+=1
            meta={'num':set_counter,'baseline':cur.copy(),'idset':cur_ids}
            known_sets.append(meta)
            return meta

        if best is None:
            set_counter+=1
            meta={'num':set_counter,'baseline':cur.copy(),'idset':cur_ids}
            known_sets.append(meta)
            return meta

        return best

    for r in runs:
        cur=ids_of(r)
        meta=choose_set(cur)
        set_no=meta['num']
        base=meta['baseline']
        base_ids=set(base.values())
        cur_ids=set(cur.values())

        if cur==base:
            labels.append(f"Set {set_no} As marked")
        elif cur_ids==base_ids:
            front=(cur['FL']==base['FR'] and cur['FR']==base['FL'])
            rear =(cur['RL']==base['RR'] and cur['RR']==base['RL'])
            if front:
                labels.append(f"Set {set_no} front sided")
            elif rear:
                labels.append(f"Set {set_no} rear sided")
            else:
                labels.append(f"Set {set_no} sided")
        else:
            newpos=[p for p in ['FL','FR','RL','RR'] if cur[p] not in base_ids]
            if set(newpos)=={'FL','FR','RL','RR'}:
                labels.append(f"Set {set_no} As marked")
            elif set(newpos)=={'FL','FR'}:
                labels.append(f"Set {set_no} with new fronts")
            elif set(newpos)=={'RL','RR'}:
                labels.append(f"Set {set_no} with new rears")
            elif len(newpos)==1:
                pos=newpos[0]
                labels.append(f"Set {set_no} with only {CORNER_LONG[pos]} new")
            else:
                parts=[f"new {CORNER_LONG[p]}" for p in ['FL','FR','RL','RR'] if p in newpos]
                labels.append(f"Set {set_no} with {', '.join(parts)}")

        set_numbers.append(set_no)

    return set_numbers, labels
