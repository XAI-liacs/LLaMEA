import numpy as np

class AdaptiveHarmonySearchWithQuantumTunneling:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 30
        self.harmony_memory_consideration_rate = 0.9
        self.adjustment_rate = 0.5
        self.tunneling_probability = 0.1
        self.positions = None
        self.best_position = None
        self.best_score = float('inf')
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
    
    def quantum_tunneling(self, position, bounds):
        # Implement quantum tunneling to escape local minima
        if np.random.rand() < self.tunneling_probability:
            lb, ub = bounds.lb, bounds.ub
            tunneled_position = np.random.uniform(lb, ub, self.dim)
            return tunneled_position
        return position
    
    def adaptive_parameters(self, evaluations):
        # Adaptively adjust parameters based on the progress
        self.harmony_memory_consideration_rate = 0.9 - 0.4 * (evaluations / self.budget)
        self.adjustment_rate = 0.5 - 0.3 * (evaluations / self.budget)
        self.tunneling_probability = 0.1 + 0.4 * (evaluations / self.budget)
    
    def generate_new_harmony(self, bounds):
        new_harmony = np.empty(self.dim)
        lb, ub = bounds.lb, bounds.ub

        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_consideration_rate:
                # Choose from harmony memory
                new_harmony[i] = self.positions[np.random.randint(self.harmony_memory_size)][i]
                if np.random.rand() < self.adjustment_rate:
                    new_harmony[i] += np.random.uniform(-1, 1) * (ub[i] - lb[i]) * 0.05
            else:
                # Random selection
                new_harmony[i] = np.random.uniform(lb[i], ub[i])

        return np.clip(new_harmony, lb, ub)

    def __call__(self, func):
        self.initialize_positions(func.bounds)
        evaluations = 0

        while evaluations < self.budget:
            new_harmony = self.generate_new_harmony(func.bounds)
            new_harmony = self.quantum_tunneling(new_harmony, func.bounds)
            new_score = func(new_harmony)
            evaluations += 1
            
            if new_score < self.best_score:
                self.best_score = new_score
                self.best_position = new_harmony
            
            worst_idx = np.argmax([func(harmony) for harmony in self.positions])
            if new_score < func(self.positions[worst_idx]):
                self.positions[worst_idx] = new_harmony
            
            self.adaptive_parameters(evaluations)