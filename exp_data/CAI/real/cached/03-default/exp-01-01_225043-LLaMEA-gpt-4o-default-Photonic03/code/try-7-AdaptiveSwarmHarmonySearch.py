import numpy as np

class AdaptiveSwarmHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 50
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par = 0.3   # Pitch Adjustment Rate
        self.harmony_memory = np.random.rand(self.harmony_memory_size, dim)
        self.best_harmony = None
        self.best_fitness = np.inf
        self.fitness_evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]
        
        # Evaluate initial harmonies
        for i in range(self.harmony_memory_size):
            if self.fitness_evaluations >= self.budget:
                break
            fitness = func(self.harmony_memory[i])
            self.fitness_evaluations += 1
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_harmony = self.harmony_memory[i].copy()
        
        while self.fitness_evaluations < self.budget:
            new_harmony = np.zeros(self.dim)

            # Generate new harmony by considering harmony memory
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    idx = np.random.randint(self.harmony_memory_size)
                    new_harmony[j] = self.harmony_memory[idx, j]
                    if np.random.rand() < self.par:
                        new_harmony[j] += np.random.normal(0, 0.01)  # Small random adjustment
                else:
                    new_harmony[j] = lower_bound[j] + np.random.rand() * (upper_bound[j] - lower_bound[j])
            
            new_harmony = np.clip(new_harmony, lower_bound, upper_bound)
            new_fitness = func(new_harmony)
            self.fitness_evaluations += 1

            # Update harmony memory if the new harmony is better
            worst_idx = np.argmax([func(h) for h in self.harmony_memory])
            if new_fitness < func(self.harmony_memory[worst_idx]):
                self.harmony_memory[worst_idx] = new_harmony

            if new_fitness < self.best_fitness:
                self.best_fitness = new_fitness
                self.best_harmony = new_harmony.copy()
        
        return self.best_harmony