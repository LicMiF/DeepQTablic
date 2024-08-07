from collections import deque
import random

class ReplayBuffer():
    def __init__(self,size):
        self._size=size
        self._buffer=deque(maxlen=size)

    def add(self,experience):
        self._buffer.append(experience)
    
    def sample(self,batchSize):
        if batchSize < len(self._buffer):
            return random.sample(self._buffer, batchSize)
        else:
            return random.sample(self._buffer, len(self._buffer))

    def __len__(self):
        return len(self.buffer)