import numpy as np

class Quantum_Harmony_Search:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 30
        self.harmony_memory = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.harmony_memory_consideration_rate = 0.95
        self.pitch_adjustment_rate = 0.1

    def initialize_harmony_memory(self, lb, ub):
        self.harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        self.scores = np.array([self.evaluate(hm) for hm in self.harmony_memory])
        self.update_best()

    def evaluate(self, solution):
        return self.func(solution)

    def random_walk(self, solution, lb, ub):
        step_size = 0.01 * (ub - lb)
        random_step = np.random.uniform(-step_size, step_size, self.dim)
        new_solution = solution + random_step
        return np.clip(new_solution, lb, ub)

    def quantum_harmony_search(self, lb, ub):
        for i in range(self.harmony_memory_size):
            new_harmony = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.harmony_memory_consideration_rate:
                    new_harmony[j] = self.harmony_memory[np.random.randint(self.harmony_memory_size)][j]
                    if np.random.rand() < self.pitch_adjustment_rate:
                        new_harmony[j] = self.random_walk(new_harmony[j], lb[j], ub[j])
                else:
                    new_harmony[j] = np.random.uniform(lb[j], ub[j])
            self.evaluate_and_update(new_harmony)

    def evaluate_and_update(self, new_harmony):
        new_score = self.evaluate(new_harmony)
        if new_score < self.best_score:
            self.best_score = new_score
            self.best_solution = new_harmony.copy()
        worst_idx = np.argmax(self.scores)
        if new_score < self.scores[worst_idx]:
            self.harmony_memory[worst_idx] = new_harmony
            self.scores[worst_idx] = new_score

    def update_best(self):
        best_idx = np.argmin(self.scores)
        self.best_solution = self.harmony_memory[best_idx].copy()
        self.best_score = self.scores[best_idx]

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_harmony_memory(lb, ub)

        while self.evaluations < self.budget:
            self.quantum_harmony_search(lb, ub)
            self.evaluations += self.harmony_memory_size

        return {'solution': self.best_solution, 'fitness': self.best_score}