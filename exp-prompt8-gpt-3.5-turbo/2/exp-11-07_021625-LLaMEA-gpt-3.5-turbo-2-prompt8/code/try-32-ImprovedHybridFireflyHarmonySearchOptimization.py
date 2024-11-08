import numpy as np

class ImprovedHybridFireflyHarmonySearchOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def firefly_harmony_search(population, max_iter):
            # Combined implementation of Firefly and Harmony Search Algorithm
            pass

        # Initialize population once
        population = np.random.uniform(-5.0, 5.0, (self.dim,))

        # Perform hybrid optimization
        for _ in range(self.budget):
            population = firefly_harmony_search(population, 10)

        # Return the best solution found
        return population