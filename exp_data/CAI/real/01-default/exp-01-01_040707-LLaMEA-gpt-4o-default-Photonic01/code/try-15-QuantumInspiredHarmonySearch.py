import numpy as np

class QuantumInspiredHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 50
        self.hmcr = 0.9
        self.par = 0.3
        self.bandwidth = 0.01
        self.position = None
        self.scores = None
        self.best_position = None
        self.best_score = float('inf')

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.harmony_memory_size, self.dim)
        self.scores = np.array([np.inf] * self.harmony_memory_size)
        
    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.harmony_memory_size):
            if scores[i] < self.scores[i]:
                self.scores[i] = scores[i]
                
            if scores[i] < self.best_score:
                self.best_score = scores[i]
                self.best_position = self.position[i]
        return scores

    def quantum_walk(self, lb, ub):
        step_size = np.random.rand(self.dim) * (ub - lb) * self.bandwidth
        return step_size * (np.random.randint(2, size=self.dim) * 2 - 1)

    def generate_new_harmony(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        new_harmony = np.zeros(self.dim)
        
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                new_harmony[i] = self.position[np.random.randint(self.harmony_memory_size), i]
                if np.random.rand() < self.par:
                    new_harmony[i] += np.random.uniform(-1, 1) * self.bandwidth * (ub[i] - lb[i])
            else:
                new_harmony[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()
        
        new_harmony += self.quantum_walk(lb, ub)
        new_harmony = np.clip(new_harmony, lb, ub)
        return new_harmony

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.harmony_memory_size
            
            new_harmony = self.generate_new_harmony(func.bounds)
            new_score = func(new_harmony)
            func_calls += 1
            
            worst_idx = np.argmax(self.scores)
            if new_score < self.scores[worst_idx]:
                self.position[worst_idx] = new_harmony
                self.scores[worst_idx] = new_score
                
                if new_score < self.best_score:
                    self.best_score = new_score
                    self.best_position = new_harmony
        
        return self.best_position, self.best_score