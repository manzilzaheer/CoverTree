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
x = np.random.rand(10,4).astype(np.float32)
print(x[0,0], x[0,1], x[1,0])

mx = np.mean(x,0)
dists = np.array([np.sqrt(np.dot(xv-mx,xv-mx)) for xv in x])
idx = np.argsort(-dists)
xs = x[idx]
t = gt()
ct = CoverTree.from_matrix(x)
b_t = gt() - t

print("Building time:", b_t, "seconds")
print("Test covering: ", ct.test_covering())

print('Generate random points')
y = np.random.rand(5,4).astype(np.float32)
print('Test Nearest Neighbour: ')
t = gt()
a = ct.nodes_containing(y)
b_t = gt() - t
print("Query time:", b_t, "seconds")
print(a)