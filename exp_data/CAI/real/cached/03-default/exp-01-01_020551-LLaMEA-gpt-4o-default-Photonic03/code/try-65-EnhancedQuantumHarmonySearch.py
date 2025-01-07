import numpy as np

class EnhancedQuantumHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.min_harmony_memory_size = max(5, dim // 2)
        self.max_harmony_memory_size = max(10, dim)
        self.harmony_memory_size = self.max_harmony_memory_size
        self.initial_hmcr = 0.9
        self.initial_par = 0.3
        self.beta_initial = 0.2
        self.beta_final = 0.05
        self.mutation_strength = 0.1
        self.adaptive_rate = 0.01
        self.memory_update_frequency = budget // 20

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        scores = np.array([func(harmony_memory[i]) for i in range(self.harmony_memory_size)])
        global_best_index = np.argmin(scores)
        global_best_position = harmony_memory[global_best_index].copy()
        evaluations = self.harmony_memory_size
        hmcr = self.initial_hmcr
        par = self.initial_par
        beta = self.beta_initial

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

                # Quantum-inspired update with dynamic step adjustment
                if np.random.rand() < beta:
                    q = np.random.normal(loc=0, scale=(1 - evaluations / self.budget))
                    new_harmony[i] = global_best_position[i] + q * (ub[i] - lb[i])

            new_score = func(new_harmony)
            evaluations += 1

            # Evolutionary replacement strategy
            if new_score < scores.max():
                worst_index = np.argmax(scores)
                harmony_memory[worst_index] = new_harmony
                scores[worst_index] = new_score

            # Strategic explorative search with hybrid memory strategy
            if evaluations < self.budget and np.random.rand() < beta:
                explorative_position = (global_best_position + new_harmony) / 2 + np.random.normal(0, self.mutation_strength, self.dim) * (ub - lb)
                explorative_position = np.clip(explorative_position, lb, ub)
                explorative_score = func(explorative_position)
                evaluations += 1
                if explorative_score < scores[global_best_index]:
                    global_best_position = explorative_position
                    scores[global_best_index] = explorative_score

            global_best_index = np.argmin(scores)
            global_best_position = harmony_memory[global_best_index].copy()

            # Adaptive parameter tuning
            hmcr = self.initial_hmcr - self.adaptive_rate * (evaluations / self.budget)
            par = self.initial_par + self.adaptive_rate * (evaluations / self.budget)
            beta = self.beta_initial - (self.beta_initial - self.beta_final) * (evaluations / self.budget)

            # Dynamic memory size adjustment
            if evaluations % self.memory_update_frequency == 0:
                self.harmony_memory_size = self.min_harmony_memory_size + (self.max_harmony_memory_size - self.min_harmony_memory_size) * (1 - evaluations / self.budget)
                self.harmony_memory_size = int(np.clip(self.harmony_memory_size, self.min_harmony_memory_size, self.max_harmony_memory_size))
                harmony_memory = np.resize(harmony_memory, (self.harmony_memory_size, self.dim))
                scores = np.resize(scores, self.harmony_memory_size)

        return global_best_position, scores[global_best_index]