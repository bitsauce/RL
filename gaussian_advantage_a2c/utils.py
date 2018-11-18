from collections import deque
import numpy as np

class FrameStack():
    def __init__(self, initial_frame, stack_size=4, preprocess_fn=None):
        # Setup initial state
        self.frame_stack = deque(maxlen=stack_size)
        initial_frame = preprocess_fn(initial_frame) if preprocess_fn else initial_frame
        for _ in range(stack_size):
            self.frame_stack.append(initial_frame)
        self.state = np.stack(self.frame_stack, axis=-1)
        self.preprocess_fn = preprocess_fn
        
    def add_frame(self, frame):
        self.frame_stack.append(self.preprocess_fn(frame))
        self.state = np.stack(self.frame_stack, axis=-1)
        
    def get_state(self):
        return self.state

class Scheduler():
    def __init__(self, initial_value, interval, decay_factor):
        self.interval = self.counter = interval
        self.decay_factor = decay_factor
        self.value_factor = 1
        self.value = initial_value
        
    def get_value(self):
        self.counter -= 1
        if self.counter < 0:
            self.counter = self.interval
            self.value *= self.decay_factor
        return self.value
        
def calculate_expected_return(rewards, gamma):
    expected_return = []
    r = 0
    for reward in rewards[::-1]: # for rewards from end to start
        r = reward + gamma * r
        expected_return.append(r)
    return expected_return[::-1] # reverse so that we get the expected return from start to end