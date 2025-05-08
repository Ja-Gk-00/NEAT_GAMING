class FitnessEvaluator:
    def __init__(self, apple_weight: float = 100.0, time_weight: float = 1.0):
        self.apple_weight = apple_weight
        self.time_weight = time_weight

    def evaluate(self, apples_eaten: int, time_steps: int) -> float:
        return apples_eaten * self.apple_weight + time_steps * self.time_weight