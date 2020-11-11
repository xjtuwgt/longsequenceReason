import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'

print(len(colors))



# from sklearn import datasets
# digits = datasets.load_digits()
# X = digits.data[:500]
# y = digits.target[:500]
#
# def tsne_show(input, out_put, ):
#
#     return
#
# tsne = TSNE(n_components=2, random_state=0)
# X_2d = tsne.fit_transform(X)
#
#
#
# target_ids = range(len(digits.target_names))
# from matplotlib import pyplot as plt
# plt.figure(figsize=(6, 5))
#
# for i, c, label in zip(target_ids, colors, digits.target_names):
#     plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
# plt.legend()
# plt.show()