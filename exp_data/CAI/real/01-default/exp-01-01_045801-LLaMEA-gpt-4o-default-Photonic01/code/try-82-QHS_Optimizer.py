import numpy as np

class QHS_Optimizer:
    def __init__(self, budget, dim, harmony_size=30, harmony_memory_consideration_rate=0.9, pitch_adjustment_rate=0.4):
        self.budget = budget
        self.dim = dim
        self.harmony_size = harmony_size
        self.harmony_memory_consideration_rate = harmony_memory_consideration_rate
        self.pitch_adjustment_rate = pitch_adjustment_rate
        self.harmonies = None
        self.best_harmony = None
        self.best_score = float('inf')
        self.evaluations = 0

    def initialize_harmonies(self, lb, ub):
        self.harmonies = np.random.uniform(lb, ub, (self.harmony_size, self.dim))
        scores = np.array([self.evaluate(harmony) for harmony in self.harmonies])
        self.best_harmony = self.harmonies[np.argmin(scores)].copy()
        self.best_score = np.min(scores)

    def evaluate(self, harmony):
        return self.func(harmony)

    def quantum_inspired_harmony(self, lb, ub):
        indices = np.random.choice(self.harmony_size, self.dim, replace=True)
        new_harmony = self.harmonies[indices, np.arange(self.dim)]
        
        for i in range(self.dim):
            if np.random.rand() > self.harmony_memory_consideration_rate:
                new_harmony[i] = np.random.uniform(lb[i], ub[i])
        
        if np.random.rand() < self.pitch_adjustment_rate:
            adjustment = np.random.uniform(-1, 1, self.dim)
            new_harmony += adjustment * (ub - lb)
            new_harmony = np.clip(new_harmony, lb, ub)
        
        quantum_superposition = np.random.uniform(-1, 1, self.dim)
        quantum_harmony = self.best_harmony + quantum_superposition * np.abs(new_harmony - self.best_harmony)
        return np.clip(quantum_harmony, lb, ub)

    def update_harmony_memory(self, new_harmony):
        new_score = self.evaluate(new_harmony)
        if new_score < self.best_score:
            self.best_score = new_score
            self.best_harmony = new_harmony.copy()

        worst_idx = np.argmax([self.evaluate(harmony) for harmony in self.harmonies])
        if new_score < self.evaluate(self.harmonies[worst_idx]):
            self.harmonies[worst_idx] = new_harmony

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_harmonies(lb, ub)

        while self.evaluations < self.budget:
            new_harmony = self.quantum_inspired_harmony(lb, ub)
            self.update_harmony_memory(new_harmony)
            self.evaluations += 1

        return {'solution': self.best_harmony, 'fitness': self.best_score}