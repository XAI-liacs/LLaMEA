import numpy as np

class PIHS_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 30
        self.harmony_search_rate = 0.8
        self.adjust_rate_range = (0.1, 0.3)
        self.harmony_memory = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.interaction_strength = 0.5

    def initialize_harmony_memory(self, lb, ub):
        self.harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        self.scores = np.array([self.evaluate(ind) for ind in self.harmony_memory])
        self.update_best()

    def evaluate(self, solution):
        return self.func(solution)

    def generate_harmony(self, lb, ub):
        harmony = np.empty(self.dim)
        for j in range(self.dim):
            if np.random.rand() < self.harmony_search_rate:
                harmony[j] = self.harmony_memory[np.random.randint(self.harmony_memory_size), j]
                if np.random.rand() < np.random.uniform(*self.adjust_rate_range):
                    harmony[j] += np.random.uniform(-1, 1) * (ub[j] - lb[j]) * self.interaction_strength
            else:
                harmony[j] = np.random.uniform(lb[j], ub[j])
        harmony = np.clip(harmony, lb, ub)
        return harmony

    def update_harmony_memory(self, new_harmony):
        new_score = self.evaluate(new_harmony)
        worst_idx = np.argmax(self.scores)
        if new_score < self.scores[worst_idx]:
            self.harmony_memory[worst_idx] = new_harmony
            self.scores[worst_idx] = new_score
            if new_score < self.best_score:
                self.best_solution = new_harmony.copy()
                self.best_score = new_score

    def particle_interaction(self):
        for i in range(self.harmony_memory_size):
            for j in range(self.dim):
                forces = np.sum(self.harmony_memory[:, j] - self.harmony_memory[i, j]) / (self.harmony_memory_size - 1)
                self.harmony_memory[i, j] += self.interaction_strength * forces
            self.harmony_memory[i] = np.clip(self.harmony_memory[i], self.func.bounds.lb, self.func.bounds.ub)
            self.scores[i] = self.evaluate(self.harmony_memory[i])
            if self.scores[i] < self.best_score:
                self.best_solution = self.harmony_memory[i].copy()
                self.best_score = self.scores[i]

    def update_best(self):
        best_idx = np.argmin(self.scores)
        self.best_solution = self.harmony_memory[best_idx].copy()
        self.best_score = self.scores[best_idx]

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_harmony_memory(lb, ub)

        while self.evaluations < self.budget:
            for _ in range(self.harmony_memory_size):
                new_harmony = self.generate_harmony(lb, ub)
                self.update_harmony_memory(new_harmony)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break
            self.particle_interaction()

        return {'solution': self.best_solution, 'fitness': self.best_score}