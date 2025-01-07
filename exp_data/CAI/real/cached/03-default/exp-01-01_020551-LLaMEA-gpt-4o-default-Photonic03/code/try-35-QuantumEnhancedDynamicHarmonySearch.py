import numpy as np

class QuantumEnhancedDynamicHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = max(10, dim)
        self.hmcr_min = 0.7
        self.hmcr_max = 0.99
        self.par_min = 0.1
        self.par_max = 0.5
        self.beta = 0.1  # Quantum learning rate
        self.mutation_strength = 0.1
        self.exploration_strength = 0.3

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        scores = np.array([func(harmony_memory[i]) for i in range(self.harmony_memory_size)])
        global_best_index = np.argmin(scores)
        global_best_position = harmony_memory[global_best_index].copy()
        evaluations = self.harmony_memory_size

        while evaluations < self.budget:
            hmcr = self.hmcr_min + (self.hmcr_max - self.hmcr_min) * (1 - evaluations / self.budget)
            par = self.par_min + (self.par_max - self.par_min) * (evaluations / self.budget)
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
                    new_harmony[i] = global_best_position[i] + q * self.exploration_strength * (ub[i] - lb[i])

            new_score = func(new_harmony)
            evaluations += 1

            # Evolutionary replacement strategy
            if new_score < scores.max():
                worst_index = np.argmax(scores)
                harmony_memory[worst_index] = new_harmony
                scores[worst_index] = new_score

            global_best_index = np.argmin(scores)
            global_best_position = harmony_memory[global_best_index].copy()

            # Quantum-enhanced local search
            if evaluations < self.budget and np.random.rand() < self.beta:
                local_search_position = global_best_position + np.random.normal(0, self.mutation_strength, self.dim) * (ub - lb)
                local_search_position = np.clip(local_search_position, lb, ub)
                local_search_score = func(local_search_position)
                evaluations += 1
                if local_search_score < scores[global_best_index]:
                    global_best_position = local_search_position
                    scores[global_best_index] = local_search_score
        
        return global_best_position, scores[global_best_index]