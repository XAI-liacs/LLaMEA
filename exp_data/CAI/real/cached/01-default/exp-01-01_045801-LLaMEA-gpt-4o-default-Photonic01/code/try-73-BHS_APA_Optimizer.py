import numpy as np

class BHS_APA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 30
        self.hmcr = 0.9  # Harmony Memory Considering Rate
        self.par_min = 0.1  # Minimum Pitch Adjustment Rate
        self.par_max = 0.5  # Maximum Pitch Adjustment Rate
        self.harmony_memory = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0

    def initialize_harmony_memory(self, lb, ub):
        self.harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        self.scores = np.array([self.evaluate(harmony) for harmony in self.harmony_memory])
        self.update_best()

    def evaluate(self, solution):
        return self.func(solution)

    def pitch_adjust(self, harmony, lb, ub):
        par = np.random.uniform(self.par_min, self.par_max)
        adjustment = np.random.uniform(-1, 1, self.dim) * par * (ub - lb)
        new_harmony = harmony + adjustment
        return np.clip(new_harmony, lb, ub)

    def improvise_new_harmony(self, lb, ub):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                idx = np.random.randint(self.harmony_memory_size)
                new_harmony[i] = self.harmony_memory[idx, i]
            else:
                new_harmony[i] = np.random.uniform(lb[i], ub[i])
        new_harmony = self.pitch_adjust(new_harmony, lb, ub)
        return new_harmony

    def update_best(self):
        best_idx = np.argmin(self.scores)
        self.best_solution = self.harmony_memory[best_idx].copy()
        self.best_score = self.scores[best_idx]

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_harmony_memory(lb, ub)

        while self.evaluations < self.budget:
            new_harmony = self.improvise_new_harmony(lb, ub)
            new_score = self.evaluate(new_harmony)
            self.evaluations += 1

            worst_idx = np.argmax(self.scores)
            if new_score < self.scores[worst_idx]:
                self.harmony_memory[worst_idx] = new_harmony
                self.scores[worst_idx] = new_score

            if new_score < self.best_score:
                self.best_score = new_score
                self.best_solution = new_harmony.copy()

        return {'solution': self.best_solution, 'fitness': self.best_score}