import numpy as np

class QuantumGuidedDynamicHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.harmony_memory_consideration_rate = 0.9
        self.pitch_adjustment_rate = 0.3
        self.bandwidth = 0.05
        self.quantum_factor_initial = 0.3
        self.quantum_factor_final = 0.1

    def quantum_adjustment(self, vector, global_best, eval_count):
        lambda_factor = (eval_count / self.budget)
        quantum_factor = self.quantum_factor_initial * (1 - lambda_factor) + self.quantum_factor_final * lambda_factor
        delta = np.random.rand(self.dim)
        new_vector = vector + quantum_factor * (global_best - vector) * delta
        return new_vector

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        harmony_memory = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        harmony_values = np.array([func(harmony) for harmony in harmony_memory])
        global_best = harmony_memory[np.argmin(harmony_values)]
        global_best_value = harmony_values.min()

        eval_count = self.population_size

        while eval_count < self.budget:
            new_harmony = np.empty(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.harmony_memory_consideration_rate:
                    selected_harmony = harmony_memory[np.random.randint(self.population_size)]
                    new_harmony[i] = selected_harmony[i]
                    if np.random.rand() < self.pitch_adjustment_rate:
                        new_harmony[i] += self.bandwidth * np.random.uniform(-1, 1)
                else:
                    new_harmony[i] = np.random.uniform(bounds[i, 0], bounds[i, 1])

            new_harmony = np.clip(new_harmony, bounds[:, 0], bounds[:, 1])
            new_harmony = self.quantum_adjustment(new_harmony, global_best, eval_count)
            new_harmony = np.clip(new_harmony, bounds[:, 0], bounds[:, 1])

            new_harmony_value = func(new_harmony)
            eval_count += 1

            if new_harmony_value < global_best_value:
                global_best = new_harmony
                global_best_value = new_harmony_value

            worst_index = np.argmax(harmony_values)
            if new_harmony_value < harmony_values[worst_index]:
                harmony_memory[worst_index] = new_harmony
                harmony_values[worst_index] = new_harmony_value

            if eval_count >= self.budget:
                break

        return global_best