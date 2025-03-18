import numpy as np

class HybridHarmonyDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.hmcr = 0.9  # Harmony Memory Considering Rate
        self.par = 0.3   # Pitch Adjusting Rate
        self.f = 0.5     # Differential weight
        self.cr = 0.9    # Crossover probability

    def __call__(self, func):
        # Initialize harmony memory
        harmony_memory = np.random.uniform(
            low=func.bounds.lb, high=func.bounds.ub, size=(self.harmony_memory_size, self.dim)
        )
        harmony_fitness = np.array([func(hm) for hm in harmony_memory])
        evaluations = self.harmony_memory_size
        
        # Optimization loop
        while evaluations < self.budget:
            # Generate new harmony
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    # Memory consideration
                    new_harmony[i] = harmony_memory[np.random.randint(self.harmony_memory_size), i]
                    if np.random.rand() < self.par:
                        # Adaptive Pitch adjustment
                        adapt_factor = 1 - (evaluations / self.budget)
                        new_harmony[i] += np.random.uniform(-1, 1) * (func.bounds.ub[i] - func.bounds.lb[i]) * 0.01 * adapt_factor
                else:
                    # Random selection
                    new_harmony[i] = np.random.uniform(func.bounds.lb[i], func.bounds.ub[i])

            new_harmony = np.clip(new_harmony, func.bounds.lb, func.bounds.ub)
            new_harmony_fitness = func(new_harmony)
            evaluations += 1

            # Differential Evolution
            if evaluations < self.budget:
                idxs = np.random.choice(self.harmony_memory_size, 3, replace=False)
                x1, x2, x3 = harmony_memory[idxs]
                self.f = 0.5 + 0.5 * (1 - evaluations / self.budget)  # Learning factor adjustment
                mutant_vector = x1 + self.f * (x2 - x3)
                trial_vector = np.where(np.random.rand(self.dim) < self.cr * (1 - evaluations / self.budget), mutant_vector, new_harmony)
                trial_vector = np.clip(trial_vector, func.bounds.lb, func.bounds.ub)
                trial_fitness = func(trial_vector)
                evaluations += 1

                # Select the best solution
                if trial_fitness < new_harmony_fitness:
                    new_harmony, new_harmony_fitness = trial_vector, trial_fitness

            # Update harmony memory if new harmony is better
            worst_idx = np.argmax(harmony_fitness)
            if new_harmony_fitness < harmony_fitness[worst_idx]:
                harmony_memory[worst_idx] = new_harmony
                harmony_fitness[worst_idx] = new_harmony_fitness

        # Return the best solution found
        best_idx = np.argmin(harmony_fitness)
        return harmony_memory[best_idx]