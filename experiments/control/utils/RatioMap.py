"""
To maintain a mapping of weights to use for tests.
"""


class RatioMap:

    def __init__(self, backward_weight, stay_weight, forward_weight):
        self.backward_ratio = backward_weight / (backward_weight + stay_weight + forward_weight)
        self.stay_ratio = stay_weight / (backward_weight + stay_weight + forward_weight)
        self.forward_ratio = forward_weight / (backward_weight + stay_weight + forward_weight)
