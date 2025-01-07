import numpy as np

class QuantumHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 30
        self.harmony_memory = None
        self.best_harmony = None
        self.best_score = float('inf')
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par = 0.3   # Pitch Adjustment Rate
        self.bandwidth = 0.05
        
    def initialize_harmony_memory(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        
    def quantum_inspired_pitch_adjustment(self, harmony):
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        adjusted_harmony = harmony + self.bandwidth * quantum_flip
        return adjusted_harmony
    
    def generate_new_harmony(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        new_harmony = np.zeros(self.dim)
        
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                # Use harmony memory
                new_harmony[i] = np.random.choice(self.harmony_memory[:, i])
                if np.random.rand() < self.par:
                    # Apply quantum-inspired pitch adjustment
                    new_harmony[i] += self.bandwidth * (2 * np.random.rand() - 1)
            else:
                # Random selection
                new_harmony[i] = np.random.uniform(lb[i], ub[i])
        
        new_harmony = np.clip(new_harmony, lb, ub)
        return new_harmony
    
    def __call__(self, func):
        self.initialize_harmony_memory(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            new_harmony = self.generate_new_harmony(func.bounds)
            new_score = func(new_harmony)
            evaluations += 1
            
            if new_score < self.best_score:
                self.best_score = new_score
                self.best_harmony = new_harmony
                
            worst_idx = np.argmax([func(h) for h in self.harmony_memory])
            if new_score < func(self.harmony_memory[worst_idx]):
                self.harmony_memory[worst_idx] = new_harmony
        
            # Apply quantum-inspired pitch adjustment globally
            for i in range(self.harmony_memory_size):
                if evaluations >= self.budget:
                    break
                adjusted_harmony = self.quantum_inspired_pitch_adjustment(self.harmony_memory[i])
                adjusted_harmony = np.clip(adjusted_harmony, func.bounds.lb, func.bounds.ub)
                adjusted_score = func(adjusted_harmony)
                evaluations += 1
                
                if adjusted_score < self.best_score:
                    self.best_score = adjusted_score
                    self.best_harmony = adjusted_harmony