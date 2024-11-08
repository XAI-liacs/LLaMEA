import numpy as np

class EnhancedHybridPSOADE_AdaptiveInertia_Improved_Optimized_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def pso_ade_improved():
            best_pos = np.random.uniform(-5.0, 5.0, self.dim)
            best_val = np.inf
            inertia_weight = 0.5

            for t in range(self.budget // 2):
                new_pos = best_pos + np.random.uniform(-1, 1, self.dim) * (np.random.uniform(-5.0, 5.0, self.dim) - best_pos)
                new_val = func(new_pos)

                if new_val < best_val:
                    best_val, best_pos = new_val, new_pos

                inertia_weight = max(0.4, 0.9 * inertia_weight * (1 - t / (self.budget // 2)))

                r = np.random.choice(self.dim)
                mutant = best_pos + 0.5 * (new_pos - best_pos) + inertia_weight * (new_pos - best_pos)
                trial = np.where(np.random.uniform(size=self.dim) < 0.5, mutant, best_pos)

                trial_val = func(trial)
                if trial_val < best_val:
                    best_pos, best_val = trial, trial_val

            return best_val

        return pso_ade_improved()