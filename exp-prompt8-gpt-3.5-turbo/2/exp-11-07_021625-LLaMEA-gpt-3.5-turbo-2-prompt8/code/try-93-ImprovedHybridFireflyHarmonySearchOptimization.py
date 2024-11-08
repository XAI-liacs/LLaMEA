import numpy as np

class ImprovedHybridFireflyHarmonySearchOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def firefly_search(population):
            # Implementation of Firefly Algorithm with optimized loop structure
            pass

        def harmony_search(population):
            # Implementation of Harmony Search Algorithm with optimized loop structure
            pass

        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (self.dim,))
        
        # Perform hybrid optimization
        for _ in range(self.budget):
            population = firefly_search(population)
            population = harmony_search(population)

        # Return the best solution found
        return population