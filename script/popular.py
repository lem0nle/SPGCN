from typing import Counter
import pandas as pd
from loguru import logger
from invest.utils import evaluate, format_metrics, load_data
import random

path = 'data/tyc/'

# load raw data
logger.info('loading data')
train = load_data(path + 'train.csv')
valid = load_data(path + 'valid.csv')
train = pd.concat([train, valid], ignore_index=True)

test = load_data(path + 'test.csv')
test_neg = load_data(path + 'test_neg.csv')

counter = Counter(train['dst_ind'])

test_all = pd.concat([test, test_neg], ignore_index=True)
user_all = test_all.groupby('src_ind')['dst_ind'].apply(list)
user_choices = test_all.groupby('src_ind')['dst_ind'].apply(
    lambda dst: set(d for _, d in sorted([(counter.get(d, 0), d) for d in dst], reverse=True)[:20])
)

src_ind = []
dst_ind = []
prediction = []
for user, cand in user_all.iteritems():
    random.shuffle(cand)
    choices = user_choices[user]
    for c in cand:
        if c in choices:
            pred = counter.get(c, 0)
        else:
            pred = 0
        src_ind.append(user)
        dst_ind.append(c)
        prediction.append(pred)
pred = pd.DataFrame({'src_ind': src_ind, 'dst_ind': dst_ind, 'prediction': prediction})
pred['prediction'] /= counter.most_common(1)[0][1]
metrics = evaluate(test, pred, top_k=[5, 10, 20])
logger.info(format_metrics(metrics))
