import numpy as np

class EnhancedQuantumGuidedEvolutionaryHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = max(10, dim)
        self.initial_hmcr = 0.9
        self.initial_par = 0.3
        self.beta = 0.1  # Quantum learning rate
        self.mutation_strength = 0.1
        self.adaptive_rate = 0.005

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        scores = np.array([func(harmony_memory[i]) for i in range(self.harmony_memory_size)])
        global_best_index = np.argmin(scores)
        global_best_position = harmony_memory[global_best_index].copy()
        evaluations = self.harmony_memory_size
        hmcr = self.initial_hmcr
        par = self.initial_par

        while evaluations < self.budget:
            new_harmony = np.empty(self.dim)
            for i in range(self.dim):
                if np.random.rand() < hmcr:
                    selected = np.random.randint(self.harmony_memory_size)
                    new_harmony[i] = harmony_memory[selected, i]
                    if np.random.rand() < par:
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
            hmcr = self.initial_hmcr - self.adaptive_rate * (evaluations / self.budget)
            par = self.initial_par + self.adaptive_rate * (evaluations / self.budget)

            # Local neighborhood search
            if evaluations < self.budget and np.random.rand() < self.beta:
                local_search_position = global_best_position + np.random.normal(0, self.mutation_strength, self.dim) * (ub - lb)
                local_search_position = np.clip(local_search_position, lb, ub)
                local_search_score = func(local_search_position)
                evaluations += 1
                if local_search_score < scores[global_best_index]:
                    global_best_position = local_search_position
                    scores[global_best_index] = local_search_score

        return global_best_position, scores[global_best_index]