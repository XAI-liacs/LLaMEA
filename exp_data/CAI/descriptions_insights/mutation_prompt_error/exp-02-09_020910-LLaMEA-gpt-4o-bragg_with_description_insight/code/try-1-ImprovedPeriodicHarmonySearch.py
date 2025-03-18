import numpy as np

class ImprovedPeriodicHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.HMS = 10  # Harmony Memory Size
        self.HMCR = 0.9  # Harmony Memory Consideration Rate
        self.PAR = 0.3  # Pitch Adjustment Rate
        self.bandwidth = 0.1  # Bandwidth for pitch adjustment
        self.evaluations = 0
    
    def adaptive_params(self, iteration, max_iterations):
        """Adaptive adjustment of Harmony Search parameters."""
        # Gradually decrease the bandwidth to fine-tune exploration over time
        self.bandwidth = 0.1 * (1 - iteration / max_iterations)
        # Gradually increase the pitch adjustment rate to focus on exploitation
        self.PAR = 0.3 + 0.7 * (iteration / max_iterations)
    
    def enforce_periodicity(self, harmony):
        """Enforce periodicity in the solution."""
        period = self.dim // 2
        harmony[:period] = harmony[period:2*period]
        return harmony
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        max_iterations = (self.budget - self.HMS)
        
        # Initialize harmony memory
        harmony_memory = np.random.uniform(lb, ub, (self.HMS, self.dim))
        harmony_fitness = np.array([func(h) for h in harmony_memory])
        self.evaluations += self.HMS

        iteration = 0
        while self.evaluations < self.budget:
            self.adaptive_params(iteration, max_iterations)
            
            # Generate new harmony
            new_harmony = np.copy(harmony_memory[np.random.randint(self.HMS)])
            for d in range(self.dim):
                if np.random.rand() < self.HMCR:
                    new_harmony[d] = harmony_memory[np.random.randint(self.HMS)][d]
                    if np.random.rand() < self.PAR:
                        new_harmony[d] += self.bandwidth * (np.random.rand() - 0.5)
                else:
                    new_harmony[d] = np.random.uniform(lb[d], ub[d])
            
            # Enforce periodicity in new harmony
            new_harmony = self.enforce_periodicity(new_harmony)

            # Evaluate new harmony
            new_fitness = func(new_harmony)
            self.evaluations += 1

            # Update harmony memory if new harmony is better
            if new_fitness > np.min(harmony_fitness):
                worst_index = np.argmin(harmony_fitness)
                harmony_memory[worst_index] = new_harmony
                harmony_fitness[worst_index] = new_fitness
            
            iteration += 1

        best_index = np.argmax(harmony_fitness)
        return harmony_memory[best_index]