import numpy as np


class Memory(object):  # stored as ( s, a, r, flag, s_ )

    def __init__(self, capacity):
        self.data_pointer = 0
        self.capacity = capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions

    def store(self, transition):
        self.data[self.data_pointer] = transition  # set the max p for new p
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def sample(self, n):
        b_idx, b_memory = np.empty((n,), dtype=np.int32), np.empty((n, self.data[0].size))
        indices = np.random.choice(self.capacity, size=n)
        j = 0
        for i in indices:
            b_memory[j, :] = self.data[i]
            j += 1

        return b_idx, b_memory, None

    def batch_update(self, tree_idx, abs_errors):
        pass
