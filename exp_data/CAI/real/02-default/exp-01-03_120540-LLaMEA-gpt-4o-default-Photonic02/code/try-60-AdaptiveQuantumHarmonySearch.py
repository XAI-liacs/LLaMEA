import numpy as np

class AdaptiveQuantumHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 30
        self.harmony_memory_consideration_rate = 0.9
        self.pitch_adjustment_rate = 0.3
        self.bandwidth = 0.05
        self.harmonies = None
        self.best_harmony = None
        self.best_score = float('inf')
    
    def initialize_harmonies(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.harmonies = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
    
    def quantum_harmony_adjustment(self, harmony, index):
        # Apply quantum-inspired pitch adjustment
        quantum_flip = np.random.choice([-1, 1], size=self.dim) * self.bandwidth
        if np.random.rand() < self.pitch_adjustment_rate:
            adjusted_harmony = harmony + quantum_flip
            return np.clip(adjusted_harmony, func.bounds.lb, func.bounds.ub)
        return harmony
    
    def generate_new_harmony(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_consideration_rate:
                random_harmony = self.harmonies[np.random.randint(self.harmony_memory_size)]
                new_harmony[i] = random_harmony[i]
            else:
                new_harmony[i] = np.random.uniform(lb[i], ub[i])
        return new_harmony
    
    def __call__(self, func):
        self.initialize_harmonies(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            new_harmony = self.generate_new_harmony(func.bounds)
            new_harmony = self.quantum_harmony_adjustment(new_harmony, evaluations)
            new_score = func(new_harmony)
            evaluations += 1
            
            if new_score < self.best_score:
                self.best_score = new_score
                self.best_harmony = new_harmony
            
            if new_score < func(self.harmonies[np.argmax([func(h) for h in self.harmonies])]):
                worst_idx = np.argmax([func(h) for h in self.harmonies])
                self.harmonies[worst_idx] = new_harmony
            
            # Adjusting pitch rate based on budget usage
            self.pitch_adjustment_rate *= (1 - evaluations / self.budget)