import numpy as np

class QuantumEnhancedAdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = max(10, dim)
        self.hmcr = 0.9
        self.par = 0.3
        self.evolution_rate = 0.1
        self.beta = 0.1
        self.mutation_strength = 0.1
        self.adaptive_factor = 0.05  # Adaptive factor for parameter tuning

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        scores = np.array([func(harmony_memory[i]) for i in range(self.harmony_memory_size)])
        global_best_index = np.argmin(scores)
        global_best_position = harmony_memory[global_best_index].copy()
        evaluations = self.harmony_memory_size

        while evaluations < self.budget:
            new_harmony = np.empty(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    selected = np.random.randint(self.harmony_memory_size)
                    new_harmony[i] = harmony_memory[selected, i]
                    if np.random.rand() < self.par:
                        new_harmony[i] += np.random.uniform(-self.mutation_strength, self.mutation_strength) * (ub[i] - lb[i])
                else:
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])

                # Quantum-inspired update
                if np.random.rand() < self.beta:
                    q = np.random.normal(loc=0, scale=1)
                    new_harmony[i] = global_best_position[i] + q * (ub[i] - lb[i])

            new_score = func(new_harmony)
            evaluations += 1

            # Evolutionary replacement strategy
            if new_score < scores.max():
                worst_index = np.argmax(scores)
                harmony_memory[worst_index] = new_harmony
                scores[worst_index] = new_score

            global_best_index = np.argmin(scores)
            global_best_position = harmony_memory[global_best_index].copy()

            # Adaptive parameter tuning
            self.beta = max(0.01, self.beta - self.adaptive_factor * (scores.min() / scores.mean()))
            self.mutation_strength = max(0.01, self.mutation_strength - self.adaptive_factor * (scores.min() / scores.mean()))

            # Stochastic evolutionary mutation
            if evaluations < self.budget and np.random.rand() < self.evolution_rate:
                mutation_indices = np.random.choice(self.harmony_memory_size, size=int(self.harmony_memory_size * self.evolution_rate), replace=False)
                for index in mutation_indices:
                    mutation_vector = np.random.normal(0, self.mutation_strength, self.dim)
                    harmony_memory[index] = np.clip(harmony_memory[index] + mutation_vector * (ub - lb), lb, ub)
                    scores[index] = func(harmony_memory[index])
                    evaluations += 1
                    if evaluations >= self.budget:
                        break

        return global_best_position, scores[global_best_index]