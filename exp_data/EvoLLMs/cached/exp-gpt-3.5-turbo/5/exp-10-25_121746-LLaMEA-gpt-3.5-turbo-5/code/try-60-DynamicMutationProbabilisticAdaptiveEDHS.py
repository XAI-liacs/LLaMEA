import numpy as np

class DynamicMutationProbabilisticAdaptiveEDHS(EnhancedEDHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = np.random.uniform(0.2, 0.4)  # Resetting mutation rate

    def __call__(self, func):
        for _ in range(self.budget):
            # Update mutation rate based on a time-varying strategy with a probability
            if np.random.rand() < 0.05:
                self.mutation_rate = np.random.uniform(0.1, 0.5)
            super().__call__(func)
        return self.get_global_best()