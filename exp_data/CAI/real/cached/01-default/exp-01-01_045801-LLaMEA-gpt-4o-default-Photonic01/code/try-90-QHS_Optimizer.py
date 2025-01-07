import numpy as np

class QHS_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 30
        self.harmony_memory = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.harmony_consideration_rate = 0.95
        self.pitch_adjustment_rate = 0.3
        self.bandwidth = 0.1

    def initialize_harmony_memory(self, lb, ub):
        self.harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        self.scores = np.array([self.evaluate(harmony) for harmony in self.harmony_memory])
        self.update_best()

    def evaluate(self, solution):
        return self.func(solution)

    def generate_new_harmony(self, lb, ub):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.harmony_consideration_rate:
                new_harmony[i] = self.harmony_memory[np.random.randint(0, self.harmony_memory_size), i]
                if np.random.rand() < self.pitch_adjustment_rate:
                    new_harmony[i] += np.random.uniform(-1, 1) * self.bandwidth
            else:
                new_harmony[i] = np.random.uniform(lb[i], ub[i])
        
        return np.clip(new_harmony, lb, ub)

    def select(self, new_harmony, new_score):
        worst_idx = np.argmax(self.scores)
        if new_score < self.scores[worst_idx]:
            self.harmony_memory[worst_idx] = new_harmony
            self.scores[worst_idx] = new_score
            if new_score < self.best_score:
                self.best_score = new_score
                self.best_solution = new_harmony.copy()

    def update_best(self):
        best_idx = np.argmin(self.scores)
        self.best_solution = self.harmony_memory[best_idx].copy()
        self.best_score = self.scores[best_idx]

    def quantum_inspired_adjustment(self):
        quantum_superposition = np.random.uniform(-1, 1, (self.harmony_memory_size, self.dim))
        adjustments = quantum_superposition * np.abs(self.harmony_memory - self.best_solution)
        self.harmony_memory = np.clip(self.harmony_memory + adjustments, self.func.bounds.lb, self.func.bounds.ub)
        self.scores = np.array([self.evaluate(harmony) for harmony in self.harmony_memory])
        self.update_best()

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_harmony_memory(lb, ub)

        while self.evaluations < self.budget:
            new_harmony = self.generate_new_harmony(lb, ub)
            new_score = self.evaluate(new_harmony)
            self.select(new_harmony, new_score)
            self.evaluations += 1
            if self.evaluations >= self.budget:
                break

            if self.evaluations % self.harmony_memory_size == 0:
                self.quantum_inspired_adjustment()
                self.evaluations += self.harmony_memory_size

        return {'solution': self.best_solution, 'fitness': self.best_score}