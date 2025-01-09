import numpy as np

class AdaptiveDifferentialHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 10
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par = 0.3   # Pitch Adjustment Rate
        self.delta = 0.1
        self.f = 0.8     # Differential weight
        self.cr = 0.9    # Crossover probability

    def __call__(self, func):
        # Initialize harmony memory
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, 
                                           (self.harmony_memory_size, self.dim))
        harmony_scores = np.array([func(h) for h in harmony_memory])
        evaluations = self.harmony_memory_size

        while evaluations < self.budget:
            # Generate a new harmony
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    # Choose from harmony memory
                    new_harmony[i] = harmony_memory[np.random.randint(0, self.harmony_memory_size), i]
                    if np.random.rand() < self.par:
                        # Pitch adjustment
                        new_harmony[i] += np.random.uniform(-self.delta, self.delta)
                        new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
                else:
                    # Random selection
                    new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
            
            # Differential mutation and crossover
            if evaluations + 3 < self.budget and self.harmony_memory_size >= 3:
                idxs = np.random.choice(self.harmony_memory_size, 3, replace=False)
                a, b, c = harmony_memory[idxs]
                mutant = np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, new_harmony)
                trial_score = func(trial)
                evaluations += 1

                # Decide whether to replace the new_harmony with trial
                new_harmony = trial if trial_score < func(new_harmony) else new_harmony
            
            # Evaluate and potentially update harmony memory
            new_score = func(new_harmony)
            evaluations += 1

            if new_score < np.max(harmony_scores):
                # Update harmony memory
                worst_idx = np.argmax(harmony_scores)
                harmony_memory[worst_idx] = new_harmony
                harmony_scores[worst_idx] = new_score

        # Return the best found solution
        best_idx = np.argmin(harmony_scores)
        return harmony_memory[best_idx]