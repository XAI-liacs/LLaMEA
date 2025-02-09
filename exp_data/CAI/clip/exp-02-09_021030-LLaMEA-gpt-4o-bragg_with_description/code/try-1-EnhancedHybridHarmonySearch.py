import numpy as np

class EnhancedHybridHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.HMCR = 0.9  # Harmony Memory Consideration Rate
        self.PAR = 0.3   # Initial Pitch Adjustment Rate
        self.bandwidth = 0.1
        self.harmony_memory_size = 10
        self.harmony_memory = None
        self.diversity_threshold = 0.05  # Threshold for diversity control

    def initialize_harmony_memory(self, bounds):
        self.harmony_memory = np.random.uniform(bounds.lb, bounds.ub, (self.harmony_memory_size, self.dim))

    def local_search(self, solution, bounds):
        local_sol = solution + np.random.uniform(-self.bandwidth, self.bandwidth, self.dim)
        return np.clip(local_sol, bounds.lb, bounds.ub)

    def adaptive_par(self, evaluations):
        return self.PAR + 0.5 * (evaluations / self.budget)  # Adaptive PAR increases over time

    def diversity_control(self):
        # Calculate diversity as the variance of the harmony memory
        diversity = np.var(self.harmony_memory, axis=0).mean()
        return diversity < self.diversity_threshold

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_harmony_memory(bounds)
        best_solution = None
        best_score = float('-inf')
        evaluations = 0

        while evaluations < self.budget:
            new_harmony = np.zeros(self.dim)
            current_PAR = self.adaptive_par(evaluations)

            for i in range(self.dim):
                if np.random.rand() < self.HMCR:
                    new_harmony[i] = self.harmony_memory[np.random.randint(self.harmony_memory_size)][i]
                    if np.random.rand() < current_PAR:
                        new_harmony[i] += self.bandwidth * (2 * np.random.rand() - 1)
                else:
                    new_harmony[i] = np.random.uniform(bounds.lb[i], bounds.ub[i])
            
            new_harmony = np.clip(new_harmony, bounds.lb, bounds.ub)
            new_harmony_score = func(new_harmony)
            evaluations += 1

            if new_harmony_score > best_score:
                best_score = new_harmony_score
                best_solution = new_harmony

            worst_idx = np.argmin([func(harmony) for harmony in self.harmony_memory])
            if new_harmony_score > func(self.harmony_memory[worst_idx]):
                self.harmony_memory[worst_idx] = new_harmony

            # Local Search Phase
            if evaluations < self.budget:
                local_solution = self.local_search(best_solution, bounds)
                local_score = func(local_solution)
                evaluations += 1
                if local_score > best_score:
                    best_solution = local_solution
                    best_score = local_score

            # Diversity Control: If diversity is low, reinitialize part of the harmony memory
            if self.diversity_control() and evaluations < self.budget:
                self.harmony_memory = np.random.uniform(bounds.lb, bounds.ub, (self.harmony_memory_size, self.dim))

        return best_solution