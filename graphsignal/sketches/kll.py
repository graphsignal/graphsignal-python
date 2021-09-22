import math
import random

# Implements https://arxiv.org/pdf/1603.05346v2.pdf


class KLLSketch:
    __slots__ = [
        '_k',
        '_c',
        '_compactors',
        '_H',
        '_size',
        '_max_size'
    ]

    def __init__(self, k=128):
        if k <= 0:
            raise ValueError("invalid k")

        self._k = k
        self._c = 2.0 / 3.0
        self._compactors = []
        self._H = 0
        self._size = 0
        self._max_size = 0

        self._grow()

    def _capacity(self, h):
        return int(math.ceil(self._k * self._c ** (self._H - h - 1))) + 1

    def _grow(self):
        self._compactors.append(Compactor())
        self._H = len(self._compactors)
        self._max_size = sum([self._capacity(h) for h in range(self._H)])

    def _compact(self):
        for h in range(len(self._compactors)):
            if len(self._compactors[h]) >= self._capacity(h):
                if h + 1 >= self._H:
                    self._grow()

                self._size -= (len(self._compactors[h]) +
                               len(self._compactors[h + 1]))
                self._compactors[h + 1].extend(self._compactors[h].compact())
                self._size += (len(self._compactors[h]) +
                               len(self._compactors[h + 1]))
                break

    def update(self, item):
        self._compactors[0].append(item)
        self._size += 1
        if self._size >= self._max_size:
            self._compact()

    def merge(self, other):
        while self._H < other._H:
            self._grow()

        for h in range(other._H):
            self._compactors[h].extend(other._compactors[h])
            self._size += len(other._compactors[h])

        while self._size >= self._max_size:
            self._compact()

    def count(self):
        count = 0
        for (h, items) in enumerate(self._compactors):
            count += len(items) * 2**h
        return count

    def distribution(self):
        weights = []
        for (h, items) in enumerate(self._compactors):
            weights.extend([(item, 2**h) for item in items])
        weights.sort()

        dist = []
        for item, weight in weights:
            if len(dist) > 0 and dist[-1][0] == item:
                dist[-1][1] += weight
            else:
                dist.append([item, weight])
        return dist

    def ranks(self):
        weights = []
        for (h, items) in enumerate(self._compactors):
            weights.extend([(item, 2**h) for item in items])

        weights.sort()

        cum_weight = 0
        pairs = []
        for (item, weight) in weights:
            cum_weight += weight
            pairs.append((item, cum_weight))
        return pairs

    def cdf(self):
        weights = []
        for (h, items) in enumerate(self._compactors):
            weights.extend([(item, 2**h) for item in items])
        total_weight = sum([weight for (item, weight) in weights])

        weights.sort()

        cum_weight = 0
        pairs = []
        for (item, weight) in weights:
            cum_weight += weight
            pairs.append((item, float(cum_weight) / float(total_weight)))

        return pairs

    def to_proto(self, proto):
        proto.k = self._k
        proto.c = self._c
        proto.H = self._H
        proto.size = self._size
        proto.max_size = self._max_size
        if isinstance(self._compactors[-1][0], (int, float)):
            proto.item_type = proto.ItemType.DOUBLE
            for (h, items) in enumerate(self._compactors):
                proto.compactors_double.add().items[:] = items
        elif isinstance(self._compactors[-1][0], str):
            proto.item_type = proto.ItemType.STRING
            for (h, items) in enumerate(self._compactors):
                proto.compactors_string.add().items[:] = items

    def from_proto(self, proto):
        self._k = proto.k
        self._c = proto.c
        self._H = proto.H
        self._size = proto.size
        self._max_size = proto.max_size
        self._compactors = []
        if proto.item_type == proto.ItemType.DOUBLE:
            for (h, compactor_proto) in enumerate(proto.compactors_double):
                compactor = Compactor()
                compactor[:] = compactor_proto.items
                self._compactors.append(compactor)
        elif proto.item_type == proto.ItemType.STRING:
            for (h, compactor_proto) in enumerate(proto.compactors_string):
                compactor = Compactor()
                compactor[:] = compactor_proto.items
                self._compactors.append(compactor)


class Compactor(list):
    def __init__(self):
        pass

    def compact(self):
        self.sort()

        offset = int(random.random() < 0.5)
        keep = self[offset::2]
        self[:] = self[-1:] if len(self) % 2 == 1 else []

        return keep
