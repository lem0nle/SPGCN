import pandas as pd
import numpy as np

gongying = pd.read_csv('data/tyc/comp_gongying_comp.csv')
gy = gongying.groupby('dst_ind')['src_ind'].apply(list)
gy_inv = gongying.groupby('src_ind')['dst_ind'].apply(list)
r = pd.DataFrame({
    'pos': np.zeros(75142, dtype=int),
    'vis': False,
})


def ans(n):
    if r.loc[n, 'vis']:
        return r.loc[n, 'pos']
    r.loc[n, 'vis'] = True
    a = 0
    try:
        s = gy[n]
    except KeyError:
        s = []
    for m in s:
        a += ans(m) + 1
    r.loc[n, 'pos'] = a
    return a


for s in range(75142):
    try:
        d = gy_inv[s]
    except KeyError:
        d = []
    if len(d) == 0:
        ans(s)


r.to_csv('data/tyc/comp_pos.csv', index=False)
