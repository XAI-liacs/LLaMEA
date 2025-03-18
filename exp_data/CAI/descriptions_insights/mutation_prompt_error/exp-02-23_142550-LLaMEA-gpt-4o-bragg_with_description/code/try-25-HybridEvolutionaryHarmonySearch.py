import numpy as np

class HybridEvolutionaryHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 30  # Number of harmonies in the memory
        self.harmony_memory = np.random.uniform(-1.0, 1.0, (self.harmony_memory_size, self.dim))
        self.best_solution = None
        self.best_score = np.inf
        self.evaluations = 0
        self.harmony_memory_consideration_rate = 0.9
        self.pitch_adjust_rate = 0.3
        self.adaptive_rate_decay = 0.99  # Decay rate for adaptive harmony memory consideration and pitch adjust rates

    def generate_new_harmony(self, lb, ub):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_consideration_rate:
                random_index = np.random.randint(self.harmony_memory_size)
                new_harmony[i] = self.harmony_memory[random_index, i]
                if np.random.rand() < self.pitch_adjust_rate:
                    new_harmony[i] += np.random.uniform(-0.01, 0.01)
            else:
                new_harmony[i] = np.random.uniform(lb[i], ub[i])
        return np.clip(new_harmony, lb, ub)

    def __call__(self, func):
        # Initialize harmony memory within given bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        self.harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        
        while self.evaluations < self.budget:
            new_harmony = self.generate_new_harmony(lb, ub)
            new_score = func(new_harmony)
            self.evaluations += 1

            worst_index = np.argmax([func(harmony) for harmony in self.harmony_memory])
            if new_score < func(self.harmony_memory[worst_index]):
                self.harmony_memory[worst_index] = new_harmony
            
            current_best_index = np.argmin([func(harmony) for harmony in self.harmony_memory])
            if func(self.harmony_memory[current_best_index]) < self.best_score:
                self.best_score = func(self.harmony_memory[current_best_index])
                self.best_solution = self.harmony_memory[current_best_index].copy()
            
            # Adapt harmony memory consideration rate and pitch adjust rate
            self.harmony_memory_consideration_rate *= self.adaptive_rate_decay
            self.pitch_adjust_rate *= self.adaptive_rate_decay

        return self.best_solution, self.best_score