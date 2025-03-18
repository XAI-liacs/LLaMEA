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
            new_harmony = np.zeros(self.dim)
            learning_rate = 0.1 + 0.9 * eval_count / self.budget  # Dynamic learning rate
            for i in range(self.dim):
                if np.random.rand() < self.phi:
                    idx = np.random.randint(self.harmony_memory_size)
                    new_harmony[i] = self.harmony_memory[idx, i]
                else:
                    new_harmony[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()

                if np.random.rand() < self.gamma:
                    adaptive_beta = 0.5 + learning_rate  # Dynamic pitch adjustment
                    new_harmony[i] += adaptive_beta * (ub[i] - lb[i]) * (np.random.rand() - 0.5)

            new_harmony = np.clip(new_harmony, lb, ub)
            new_value = func(new_harmony)
            eval_count += 1

            worst_idx = np.argmax(func_values)
            if new_value < func_values[worst_idx]:
                self.harmony_memory[worst_idx] = (1 - learning_rate) * self.harmony_memory[worst_idx] + learning_rate * new_harmony
                func_values[worst_idx] = new_value
                self.phi = min(1.0, self.phi + 0.01)

            self.gamma *= 0.98  # Adjusted decay strategy for gamma
            self.beta *= 0.97  # Adjusted adaptive beta decay

        best_idx = np.argmin(func_values)
        return self.harmony_memory[best_idx], func_values[best_idx]