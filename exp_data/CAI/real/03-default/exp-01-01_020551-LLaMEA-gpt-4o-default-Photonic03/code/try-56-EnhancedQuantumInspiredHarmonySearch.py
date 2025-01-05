import numpy as np

class EnhancedQuantumInspiredHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.min_harmony_memory_size = max(5, dim // 2)
        self.max_harmony_memory_size = max(10, dim)
        self.initial_hmcr = 0.95  # Higher harmony memory consideration rate
        self.initial_par = 0.3
        self.beta = 0.25  # Increased quantum learning rate for better exploration
        self.mutation_strength = 0.1
        self.adaptive_rate = 0.015  # Increased adaptive rate for dynamic adjustments
        self.memory_update_frequency = budget // 15  # More frequent memory updates
        self.elite_memory_size = 3  # Elite memory size to store best solutions
        self.elite_memory = []

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

                # Quantum-inspired update with elite memory consideration
                if np.random.rand() < self.beta:
                    elite_choice = np.random.choice(self.elite_memory) if self.elite_memory else global_best_position
                    q = np.random.normal(loc=0, scale=0.3)  # More controlled quantum step
                    new_harmony[i] = elite_choice[i] + q * (ub[i] - lb[i])

            new_score = func(new_harmony)
            evaluations += 1

            # Update elite memory
            if len(self.elite_memory) < self.elite_memory_size:
                self.elite_memory.append(new_harmony)
            else:
                worst_elite_index = np.argmax([func(e) for e in self.elite_memory])
                if new_score < func(self.elite_memory[worst_elite_index]):
                    self.elite_memory[worst_elite_index] = new_harmony

            # Evolutionary replacement strategy
            if new_score < scores.max():
                worst_index = np.argmax(scores)
                harmony_memory[worst_index] = new_harmony
                scores[worst_index] = new_score

            # Strategic explorative search with quantum leap
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
                harmony_memory_size = self.min_harmony_memory_size + (self.max_harmony_memory_size - self.min_harmony_memory_size) * (1 - evaluations / self.budget)
                harmony_memory_size = int(np.clip(harmony_memory_size, self.min_harmony_memory_size, self.max_harmony_memory_size))
                harmony_memory = np.resize(harmony_memory, (harmony_memory_size, self.dim))
                scores = np.resize(scores, harmony_memory_size)

        return global_best_position, scores[global_best_index]