from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from pandas import read_csv

import matplotlib.pyplot as plt
import numpy as np

def load_data(filename):
    data = read_csv(filename, header=1)
    dataset = data.values
    X = dataset[:,:-1]
    y = dataset[:,-1]
    X = X.astype(float)
    y = y.reshape((len(y),1))
    return X, y
X, y = load_data("color_ratio_averages_new.csv")
print(X)
print(y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = {
    "red": (1,0,0),
    "blue": (0,0,1),
    "green":( 0,1,0),
    "yellow": (1,1,0),
    "white": (1,1,1),
    "orange": (1,0.5,0)
}
for i, color in enumerate(y):
     ax.scatter([X[i][0]],[X[i][1]],[X[i][2]], c=[colors[color[0]]], marker='o')

# xs = X[:,0]
# ys = X[:,1]
# zs = X[:,2]

# m = 'o'

# ax.scatter(xs, ys, zs, marker=m)

ax.set_xlabel('Red/Green')
ax.set_ylabel('Red/Blue')
ax.set_zlabel('Green/Blue')

plt.show()