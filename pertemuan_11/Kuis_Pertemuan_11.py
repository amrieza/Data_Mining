import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

siswa = np.array([1, 2, 3, 4, 5])
dtw = np.array([2, 3, 4, 1, 3])
dtt = np.array([4, 4, 3, 5, 2])
dmt = np.array([4, 3, 2, 4, 1])
ddb = np.array([3, 5, 5, 2, 3])

data = np.array(list(zip(dtw, dtt, dmt, ddb)))

complete_linkage = linkage(data, method='complete', metric='cityblock')

average_linkage = linkage(data, method='average', metric='cityblock')

plt.figure(figsize=(10, 5))
dendrogram(complete_linkage, labels=siswa, orientation='top', distance_sort='descending')
plt.title('Dendrogram Complete Linkage')
plt.show()

plt.figure(figsize=(10, 5))
dendrogram(average_linkage, labels=siswa, orientation='top', distance_sort='descending')
plt.title('Dendrogram Average Linkage')
plt.show()
