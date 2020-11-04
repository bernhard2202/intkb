import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import axes3d


parser = argparse.ArgumentParser()

parser.add_argument('--graph_type', type=str, default='before', help='before/after rankerNet [default=before]')

args = parser.parse_args()

df1 = pd.read_csv('./before_rankerNet.csv')
df2 = pd.read_csv('./after_rankerNet.csv')

# before_rankerNet graph
if (args.graph_type == 'before'):
    df = df1.copy()
else:
    df = df2.copy()
# after_rankerNet graph
# df = df2.copy()

X, Y, Z = df['null_odd'], df['span_score'], df['H@1']
O = df['doc_left']
fig = plt.figure(figsize=(30,15))
ax = fig.add_subplot(121, projection='3d')
ax.plot_trisurf(X, Y, Z, antialiased=False)
ax.set_xlabel('null_odd')
ax.set_ylabel('span_score')
ax.set_zlabel('H@1')
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_trisurf(X, Y, O, antialiased=False)
ax2.set_xlabel('null_odd')
ax2.set_ylabel('span_score')
ax2.set_zlabel('Docs_left')
plt.show()

