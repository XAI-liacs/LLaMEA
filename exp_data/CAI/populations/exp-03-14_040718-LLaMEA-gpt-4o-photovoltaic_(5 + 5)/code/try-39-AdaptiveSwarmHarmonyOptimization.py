import numpy as np

class AdaptiveSwarmHarmonyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * dim  # Adaptive based on dimension
        self.harmony_memory_size = max(5, dim)  # Size of harmony memory
        self.harmony_memory = np.random.rand(self.harmony_memory_size, dim)
        self.phi = 0.5  # Harmony memory consideration rate
        self.beta = 1.0  # Parameter to control randomization
        self.gamma = 0.5  # Parameter to control local search

    def __call__(self, func):
        # Initialize harmony memory with random solutions
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        self.harmony_memory = lb + (ub - lb) * np.random.rand(self.harmony_memory_size, self.dim)
        func_values = np.array([func(hm) for hm in self.harmony_memory])
        eval_count = self.harmony_memory_size

        while eval_count < self.budget:
            # Generate a new harmony vector
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                # Introduced a nonlinear decay for phi
                self.phi = 0.5 * (1 - (eval_count / self.budget)**2)
                if np.random.rand() < self.phi:
                    idx = np.random.randint(self.harmony_memory_size)
                    new_harmony[i] = self.harmony_memory[idx, i]
                else:
                    new_harmony[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()

                # Adjust new harmony using pitch adjustment
                # Dynamic adjustment for gamma
                self.gamma = 0.5 * (1 - (eval_count / self.budget)**0.5)
                if np.random.rand() < self.gamma:
                    self.beta = np.random.uniform(0.5, 1.5)  # Adaptive pitch adjustment
                    new_harmony[i] += self.beta * (ub[i] - lb[i]) * (np.random.rand() - 0.5)

            # Ensure new harmony is within bounds
            new_harmony = np.clip(new_harmony, lb, ub)

            # Evaluate the new harmony
            new_value = func(new_harmony)
            eval_count += 1

            # Update harmony memory if the new solution is better
            worst_idx = np.argmax(func_values)
            if new_value < func_values[worst_idx]:
                self.harmony_memory[worst_idx] = new_harmony
                func_values[worst_idx] = new_value

        # Return the best found solution and its value
        best_idx = np.argmin(func_values)
        return self.harmony_memory[best_idx], func_values[best_idx]