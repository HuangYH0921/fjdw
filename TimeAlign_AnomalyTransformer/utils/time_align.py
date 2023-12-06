from itertools import combinations
from copy import deepcopy
import pandas as pd
import numpy as np

def align_period(ts1, ts2):
    td1, td2 = ts1.shape[0], ts2.shape[0]
    if td1 == td2:
        return ts1, ts2
    switch = 0
    if td1 < td2:
        ts1, ts2 = ts2, ts1
        td1, td2 = td2, td1
        switch = 1
    insert_pos = np.round(np.linspace(0, ts2.shape[0], ts1.shape[0]-ts2.shape[0]+2)[1:-1]).astype(int)
    for i in insert_pos:
        ts2 = pd.concat((ts2, pd.DataFrame(ts2.iloc[i-1:i+1].mean()).T))
    ts2 = ts2.sort_values(by='timestamp')
    if switch:
        ts1, ts2 = ts2, ts1
    return ts1, ts2

def align_2yrs(ts1, ts2, tp1, tp2):
    assert len(tp1)==len(tp2)
    ts1s, ts2s = [], []
    for i in range(len(tp1)-1):
        tmp1 = ts1[(pd.to_datetime(ts1['timestamp'])>=pd.to_datetime(tp1[i]))&(pd.to_datetime(ts1['timestamp'])<pd.to_datetime(tp1[i+1]))]
        tmp2 = ts2[(pd.to_datetime(ts2['timestamp'])>=pd.to_datetime(tp2[i]))&(pd.to_datetime(ts2['timestamp'])<pd.to_datetime(tp2[i+1]))]
        res = align_period(tmp1, tmp2)
        ts1s.append(res[0])
        ts2s.append(res[1])
    ts1, ts2 = pd.concat(ts1s), pd.concat(ts2s)
    return ts1, ts2

def align_mulyrs(tss, tps):
    # tss_new = deepcopy(tss)
    for ind in combinations(range(len(tss)), 2):
        t1, tp1 = tss[ind[0]], tps[ind[0]]
        t2, tp2 = tss[ind[1]], tps[ind[1]]
        t1, t2 = align_2yrs(t1, t2, tp1, tp2)
        tss[ind[0]], tss[ind[1]] = t1, t2
    for i in range(len(tss)):
        tss[i] = tss[i].reset_index()
        tss[i]['label'] = tss[i]['label']+0.5
        tss[i]['label'] = tss[i]['label'].astype(int)
    return tss

def merge_ts(*tss):
    return pd.concat(tss)
