import numpy as np

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = max(10, min(50, budget // 10))
        self.harmonies = None
        self.harmony_memory_consideration_rate = 0.9
        self.pitch_adjustment_rate = 0.3
        self.dynamic_adjustment_rate = 0.01
        self.best_harmony = None
        self.best_fitness = float('inf')
        
    def initialize_harmonies(self, lb, ub):
        self.harmonies = lb + (ub - lb) * np.random.rand(self.harmony_memory_size, self.dim)
        
    def evaluate_harmonies(self, func):
        fitness = np.array([func(h) for h in self.harmonies])
        for i, f in enumerate(fitness):
            if f < self.best_fitness:
                self.best_fitness = f
                self.best_harmony = self.harmonies[i]
        return fitness
    
    def generate_new_harmony(self, lb, ub):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_consideration_rate:
                new_harmony[i] = self.harmonies[np.random.randint(self.harmony_memory_size)][i]
                if np.random.rand() < self.pitch_adjustment_rate:
                    new_harmony[i] += self.dynamic_adjustment_rate * (ub[i] - lb[i]) * np.random.uniform(-1, 1)
            else:
                new_harmony[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()
        new_harmony = np.clip(new_harmony, lb, ub)
        return new_harmony
    
    def update_harmony_memory(self, new_harmony, new_fitness):
        worst_index = np.argmax([func(h) for h in self.harmonies])
        if new_fitness < self.best_fitness:
            self.harmonies[worst_index] = new_harmony
            self.best_fitness = new_fitness
            self.best_harmony = new_harmony
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_harmonies(lb, ub)
        evaluations = self.harmony_memory_size
        
        while evaluations < self.budget:
            new_harmony = self.generate_new_harmony(lb, ub)
            new_fitness = func(new_harmony)
            self.update_harmony_memory(new_harmony, new_fitness)
            evaluations += 1
        
        return self.best_harmony, self.best_fitness