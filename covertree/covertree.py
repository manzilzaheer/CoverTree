import covertreec

class CoverTree(object):
    """CoverTree Class"""
    def __init__(self, this):
        self.this = this

    def __del__(self):
        covertreec.delete(self.this)

    @classmethod
    def from_matrix(cls, points):
        ptr = covertreec.new(points)
        return cls(ptr)

    def insert(self, point):
        return covertreec.insert(self.this, point)

    def remove(self, point):
        return covertreec.remove(self.this, point)

    def NearestNeighbour(self, points):
        return covertreec.NearestNeighbour(self.this, points)

    def kNearestNeighbours(self, points, k=10):
        return covertreec.kNearestNeighbours(self.this, points, k)

    def display(self):
        return covertreec.display(self.this)

    def test_covering(self):
        return covertreec.test_covering(self.this)

