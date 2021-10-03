import torch as th
from src.utils import load_data, dump_result, evaluate
from src.model import LightGCN


# load raw data
print('loading data')
train = load_data('..')
test = load_data('..')

# build data loader

# build model
print('building model')
params = {}
model = LightGCN(params)

# train and save model
print('training...')
model.fit(train)
model.save('LightGCN.snapshot')

print('training finished')

# predict
print('predicting...')
pred = model.predict(test)
dump_result(pred, '..')

# evaluate
metrics = evaluate(test, pred)
print(metrics)
