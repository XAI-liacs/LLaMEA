import numpy as np

class EnhancedQuantumInspiredDynamicHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.min_harmony_memory_size = max(5, dim // 2)
        self.max_harmony_memory_size = max(10, dim)
        self.initial_hmcr = 0.9
        self.initial_par = 0.3
        self.beta = 0.3  # Enhanced quantum learning rate for better exploration
        self.mutation_strength = 0.1
        self.adaptive_rate = 0.02  # Increased adaptive rate for more dynamic tuning
        self.memory_update_frequency = budget // 15  # More frequent updates for reactive memory size

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory_size = self.max_harmony_memory_size
        harmony_memory = np.random.uniform(lb, ub, (harmony_memory_size, self.dim))
        scores = np.array([func(harmony_memory[i]) for i in range(harmony_memory_size)])
        global_best_index = np.argmin(scores)
        global_best_position = harmony_memory[global_best_index].copy()
        evaluations = harmony_memory_size
        hmcr = self.initial_hmcr
        par = self.initial_par

        while evaluations < self.budget:
            new_harmony = np.empty(self.dim)
            for i in range(self.dim):
                if np.random.rand() < hmcr:
                    selected = np.random.randint(harmony_memory_size)
                    new_harmony[i] = harmony_memory[selected, i]
                    if np.random.rand() < par:
                        new_harmony[i] += np.random.uniform(-self.mutation_strength, self.mutation_strength) * (ub[i] - lb[i])
                else:
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])

                # Enhanced quantum-inspired update
                if np.random.rand() < self.beta:
                    q = np.random.normal(loc=0, scale=0.3)  # More precise quantum step
                    new_harmony[i] = global_best_position[i] + q * (ub[i] - lb[i])

            new_score = func(new_harmony)
            evaluations += 1

            # Updated evolutionary replacement strategy
            if new_score < scores.max():
                worst_index = np.argmax(scores)
                harmony_memory[worst_index] = new_harmony
                scores[worst_index] = new_score

            # Advanced explorative search with quantum leap
            if evaluations < self.budget and np.random.rand() < self.beta:
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

            # Dynamic memory size adjustment
            if evaluations % self.memory_update_frequency == 0:
                harmony_memory_size = int(self.min_harmony_memory_size + (self.max_harmony_memory_size - self.min_harmony_memory_size) * (1 - evaluations / self.budget))
                harmony_memory_size = int(np.clip(harmony_memory_size, self.min_harmony_memory_size, self.max_harmony_memory_size))
                if harmony_memory_size < len(harmony_memory):
                    harmony_memory = harmony_memory[:harmony_memory_size]
                    scores = scores[:harmony_memory_size]
                else:
                    additional_memory = np.random.uniform(lb, ub, (harmony_memory_size - len(harmony_memory), self.dim))
                    harmony_memory = np.vstack((harmony_memory, additional_memory))
                    additional_scores = np.array([func(additional_memory[i]) for i in range(additional_memory.shape[0])])
                    scores = np.concatenate((scores, additional_scores))
                    evaluations += additional_memory.shape[0]

        return global_best_position, scores[global_best_index]