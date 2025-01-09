import numpy as np

class HarmonySearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def random_solution():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        def adjust_value(value):
            return max(self.lower_bound, min(self.upper_bound, value))

        def objective_function(solution):
            return func(solution)

        harmonies = [random_solution() for _ in range(self.budget)]
        best_solution = min(harmonies, key=objective_function)

        for _ in range(self.budget):
            new_harmony = [adjust_value(np.random.normal(np.mean([h[d] for h in harmonies]), np.std([h[d] for h in harmonies])) 
                           for d in range(self.dim)]
            if objective_function(new_harmony) < objective_function(best_solution):
                best_solution = new_harmony
            harmonies[np.argmax([objective_function(h) for h in harmonies])] = new_harmony

        return best_solution