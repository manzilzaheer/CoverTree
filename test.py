# Copyright (c) 2017 Manzil Zaheer All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import numpy as np
import scipy as sc
from covertree.covertree import CoverTree
from sklearn.neighbors import NearestNeighbors

gt = time.time
np.random.seed(seed=3)

print('Building cover tree')
x = np.random.rand(500000,128).astype(np.float32)
with open('train_data.bin', 'wb') as f:
    np.array(x.shape, dtype='int32').tofile(f)
    x.tofile(f)
print(x[0,0], x[0,1], x[1,0])

mx = np.mean(x,0)
dists = np.array([np.sqrt(np.dot(xv-mx,xv-mx)) for xv in x])
idx = np.argsort(-dists)
xs = x[idx]
#print sc.spatial.distance.squareform(sc.spatial.distance.pdist(x, 'euclidean'))
t = gt()
ct = CoverTree.from_matrix(x)
b_t = gt() - t
#ct.display()
print("Building time:", b_t, "seconds")

print("Test covering: ", ct.test_covering())

print('Generate random points')
y = np.random.rand(5000,128).astype(np.float32)
with open('test_data.bin', 'wb') as f:
    np.array(y.shape, dtype='int32').tofile(f)
    y.tofile(f)

print('Test Nearest Neighbour: ')
t = gt()
a = ct.NearestNeighbour(y)
b_t = gt() - t
print("Query time:", b_t, "seconds")
nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(xs)
distances, indices = nbrs.kneighbors(y)
b = np.squeeze(xs[indices])
if np.all(a==b):
    print("Test for Nearest Neighbour passed")
else:
    print("Test for Nearest Neighbour failed")

print('Test k-Nearest Neighbours (k=2): ')
t = gt()
a = ct.kNearestNeighbours(y,2)
b_t = gt() - t
print("Query time:", b_t, "seconds")
nbrs = NearestNeighbors(n_neighbors=2, algorithm='brute').fit(xs)
distances, indices = nbrs.kneighbors(y)
if np.all(a==xs[indices]):
    print("Test for k-Nearest Neighbours passed")
else:
    print("Test for k-Nearest Neighbours failed")

print('Test delete: ')
x2 = np.vstack((xs[:indices[0,0]], xs[indices[0,0]+1:]))
dels = ct.remove(xs[indices[0,0]])
print('Point deleted: ', dels)
a = ct.NearestNeighbour(y)
nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(x2)
distances, indices = nbrs.kneighbors(y)
b = np.squeeze(x2[indices])
if np.all(a==b):
	print("Test for delete passed")
else:
	print("Test for delete failed")
