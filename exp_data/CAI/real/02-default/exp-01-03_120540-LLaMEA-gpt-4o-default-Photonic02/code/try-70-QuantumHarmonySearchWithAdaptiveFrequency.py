import numpy as np

class QuantumHarmonySearchWithAdaptiveFrequency:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 30
        self.harmony_memory = None
        self.best_harmony = None
        self.best_score = float('inf')
        self.harmony_consideration_rate = 0.9
        self.adaptive_pitch_adjustment_rate = 0.5
        self.frequency_range = 0.01

    def initialize_harmony_memory(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
    
    def quantum_inspired_movement(self, harmony):
        quantum_shift = np.random.uniform(-self.frequency_range, self.frequency_range, self.dim)
        return harmony + quantum_shift
    
    def create_new_harmony(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        new_harmony = np.copy(self.harmony_memory[np.random.choice(self.harmony_memory_size)])
        
        for i in range(self.dim):
            if np.random.rand() < self.harmony_consideration_rate:
                new_harmony[i] = self.harmony_memory[np.random.choice(self.harmony_memory_size)][i]
            if np.random.rand() < self.adaptive_pitch_adjustment_rate:
                new_harmony[i] = new_harmony[i] + np.random.uniform(-self.frequency_range, self.frequency_range)

        new_harmony = np.clip(new_harmony, lb, ub)
        return new_harmony, func(new_harmony)

    def __call__(self, func):
        self.initialize_harmony_memory(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            new_harmony, new_score = self.create_new_harmony(func)
            evaluations += 1
            
            if new_score < self.best_score:
                self.best_score = new_score
                self.best_harmony = new_harmony
            
            worst_idx = np.argmax([func(h) for h in self.harmony_memory])
            if new_score < func(self.harmony_memory[worst_idx]):
                self.harmony_memory[worst_idx] = new_harmony
            
            # Adaptive frequency adjustment
            self.adaptive_pitch_adjustment_rate = 0.5 * (1 - evaluations / self.budget)
        
        # Final quantum-inspired tuning
        for i in range(self.harmony_memory_size):
            if evaluations >= self.budget:
                break
            
            quantum_harmony = self.quantum_inspired_movement(self.harmony_memory[i])
            quantum_harmony = np.clip(quantum_harmony, func.bounds.lb, func.bounds.ub)
            quantum_score = func(quantum_harmony)
            evaluations += 1
            
            if quantum_score < self.best_score:
                self.best_score = quantum_score
                self.best_harmony = quantum_harmony