import numpy as np

class HS_AHM_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 30
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par_min = 0.1  # Minimum Pitch Adjustment Rate
        self.par_max = 0.5  # Maximum Pitch Adjustment Rate
        self.bw = 0.02  # Bandwidth for pitch adjustment
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)
        
        # Initialize Harmony Memory (HM)
        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        harmony_values = np.array([func(h) for h in harmony_memory])
        evaluations = self.harmony_memory_size
        
        while evaluations < self.budget:
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    # Memory consideration
                    idx = np.random.randint(0, self.harmony_memory_size)
                    new_harmony[i] = harmony_memory[idx, i]
                    if np.random.rand() < self.par_min + (self.par_max - self.par_min) * (evaluations / self.budget):
                        # Pitch adjustment
                        new_harmony[i] += self.bw * np.random.uniform(-1, 1)
                else:
                    # Random selection
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])
            new_harmony = np.clip(new_harmony, lb, ub)
            
            new_value = func(new_harmony)
            evaluations += 1
            
            # Update harmony memory if new harmony is better
            if new_value < np.max(harmony_values):
                worst_idx = np.argmax(harmony_values)
                harmony_memory[worst_idx] = new_harmony
                harmony_values[worst_idx] = new_value
        
        best_idx = np.argmin(harmony_values)
        return harmony_memory[best_idx], harmony_values[best_idx]