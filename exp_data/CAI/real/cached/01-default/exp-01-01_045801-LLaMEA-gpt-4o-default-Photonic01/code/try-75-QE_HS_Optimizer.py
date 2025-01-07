import numpy as np

class QE_HS_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 30
        self.HMCR = 0.7  # Harmony Memory Consideration Rate
        self.PAR_min = 0.1  # Pitch Adjusting Rate
        self.PAR_max = 0.9
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

    def improvise_new_harmony(self, lb, ub):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.HMCR:
                new_harmony[i] = np.random.choice(self.harmony_memory[:, i])
                if np.random.rand() < self.dynamic_PAR():
                    new_harmony[i] += np.random.uniform(-1, 1) * (ub[i] - lb[i])
                    new_harmony[i] = np.clip(new_harmony[i], lb[i], ub[i])
            else:
                new_harmony[i] = np.random.uniform(lb[i], ub[i])
        return new_harmony

    def dynamic_PAR(self):
        return np.random.uniform(self.PAR_min, self.PAR_max)

    def quantum_mutation(self, harmony):
        quantum_shift = np.random.uniform(-1, 1, self.dim)
        quantum_harmony = self.best_solution + quantum_shift * np.abs(harmony - self.best_solution)
        return np.clip(quantum_harmony, self.func.bounds.lb, self.func.bounds.ub)

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

            if new_score < self.best_score:
                self.best_solution = new_harmony.copy()
                self.best_score = new_score

            worst_idx = np.argmax(self.scores)
            if new_score < self.scores[worst_idx]:
                self.harmony_memory[worst_idx] = new_harmony
                self.scores[worst_idx] = new_score

            if self.evaluations < self.budget:
                quantum_harmony = self.quantum_mutation(new_harmony)
                quantum_score = self.evaluate(quantum_harmony)
                self.evaluations += 1
                if quantum_score < self.best_score:
                    self.best_solution = quantum_harmony.copy()
                    self.best_score = quantum_score
                
                worst_idx = np.argmax(self.scores)
                if quantum_score < self.scores[worst_idx]:
                    self.harmony_memory[worst_idx] = quantum_harmony
                    self.scores[worst_idx] = quantum_score

        return {'solution': self.best_solution, 'fitness': self.best_score}