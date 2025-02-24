import numpy as np
from scipy.optimize import minimize

class AdaptiveHarmonySearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = None
        self.ub = None

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        harmony_memory_size = 10 + 3 * self.dim
        harmony_memory = self.initialize_population(harmony_memory_size)
        harmony_memory_fitness = np.array([func(harmony) for harmony in harmony_memory])
        eval_count = harmony_memory_size

        hmcr = 0.9  # Harmony memory consideration rate
        par_min, par_max = 0.1, 0.5  # Pitch adjustment rate range
        bw = 0.01  # Bandwidth for pitch adjustment

        while eval_count < self.budget:
            if eval_count >= self.budget:
                break

            # Generate a new harmony
            new_harmony = np.empty(self.dim)
            for i in range(self.dim):
                if np.random.rand() < hmcr:
                    idx = np.random.randint(harmony_memory_size)
                    new_harmony[i] = harmony_memory[idx][i]
                    if np.random.rand() < par_min + (par_max - par_min) * (eval_count / self.budget):
                        new_harmony[i] += (2 * np.random.rand() - 1) * bw
                else:
                    new_harmony[i] = np.random.uniform(self.lb[i], self.ub[i])

            new_harmony = np.clip(new_harmony, self.lb, self.ub)
            new_harmony = self.enforce_periodicity(new_harmony)

            # Evaluate the new harmony
            new_fitness = func(new_harmony)
            eval_count += 1

            # Update harmony memory if the new harmony is better
            worst_idx = np.argmax(harmony_memory_fitness)
            if new_fitness < harmony_memory_fitness[worst_idx]:
                harmony_memory[worst_idx] = new_harmony
                harmony_memory_fitness[worst_idx] = new_fitness

            # Local optimization with L-BFGS-B
            if eval_count < self.budget:
                best_idx = np.argmin(harmony_memory_fitness)
                result = minimize(func, harmony_memory[best_idx], bounds=list(zip(self.lb, self.ub)), method='L-BFGS-B')
                eval_count += result.nfev
                if result.fun < harmony_memory_fitness[best_idx]:
                    harmony_memory[best_idx] = result.x
                    harmony_memory_fitness[best_idx] = result.fun

        best_idx = np.argmin(harmony_memory_fitness)
        return harmony_memory[best_idx]

    def initialize_population(self, size):
        return np.random.uniform(self.lb, self.ub, (size, self.dim))

    def enforce_periodicity(self, vector):
        period = 2
        num_periods = self.dim // period
        for i in range(num_periods):
            mean_value = np.mean(vector[i*period:(i+1)*period])
            vector[i*period:(i+1)*period] = mean_value
        return vector