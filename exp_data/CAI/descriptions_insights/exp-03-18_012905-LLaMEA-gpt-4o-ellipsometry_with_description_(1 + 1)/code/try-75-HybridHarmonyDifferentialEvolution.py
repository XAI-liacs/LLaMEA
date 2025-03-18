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
        
        while evaluations < self.budget:
            current_harmony_memory_size = max(5, self.harmony_memory_size - int((evaluations / self.budget) * 5))
            
            # Generate new harmony
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[i] = harmony_memory[np.random.randint(current_harmony_memory_size), i]
                    if np.random.rand() < self.par:
                        adapt_factor = 1 - (evaluations / self.budget)
                        new_harmony[i] += np.random.uniform(-1, 1) * (func.bounds.ub[i] - func.bounds.lb[i]) * 0.01 * adapt_factor
                else:
                    new_harmony[i] = np.random.uniform(func.bounds.lb[i], func.bounds.ub[i])

            new_harmony = np.clip(new_harmony, func.bounds.lb, func.bounds.ub)
            new_harmony_fitness = func(new_harmony)
            evaluations += 1

            if evaluations < self.budget:
                idxs = np.random.choice(current_harmony_memory_size, 3, replace=False)
                x1, x2, x3 = harmony_memory[idxs]
                self.f = 0.5 + 0.5 * np.random.rand() * (1 - evaluations / self.budget)  # Randomize learning factor
                mutant_vector = x1 + self.f * (x2 - x3)
                trial_vector = np.where(np.random.rand(self.dim) < self.cr, mutant_vector, new_harmony)
                trial_vector = np.clip(trial_vector, func.bounds.lb, func.bounds.ub)
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < new_harmony_fitness:
                    new_harmony, new_harmony_fitness = trial_vector, trial_fitness

            tournament_indices = np.random.choice(current_harmony_memory_size, 2, replace=False)
            worst_idx = tournament_indices[np.argmax(harmony_fitness[tournament_indices])]  # Tournament selection
            if new_harmony_fitness < harmony_fitness[worst_idx]:
                harmony_memory[worst_idx] = new_harmony
                harmony_fitness[worst_idx] = new_harmony_fitness

        best_idx = np.argmin(harmony_fitness)
        return harmony_memory[best_idx]