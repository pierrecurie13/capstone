import pandas as pd

df = pd.read_csv('data/training2017/REFERENCE.csv', header=None, names = ['name','label'])
df = df[df.label != 'O']
test = df.sample(frac=.2)
trainIdx = list(set(df.index) - set(test.index))
train = df.loc[trainIdx]