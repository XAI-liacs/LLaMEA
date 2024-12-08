import numpy as np

class DynamicInertiaPSOADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def pso_ade():
            best_pos = np.random.uniform(-5.0, 5.0, self.dim)
            best_val = np.inf
            inertia_weight = 0.5  # Initial inertia weight

            for _ in range(self.budget):
                # PSO update with dynamic inertia weight
                rand_uniform = np.random.uniform(size=self.dim)
                new_pos = best_pos + inertia_weight * (best_pos - best_pos) + inertia_weight * (np.random.uniform(-5.0, 5.0, self.dim) - best_pos)
                new_val = func(new_pos)

                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos

                # ADE update with optimized mutation strategy
                r1, r2, r3 = np.random.choice(range(self.dim), 3, replace=False)
                mutant = best_pos + 0.5 * (best_pos - best_pos) + 0.5 * (np.random.uniform(-5.0, 5.0, self.dim) - best_pos)
                trial = np.where(np.random.uniform(size=self.dim) < 0.5, mutant, best_pos)

                if func(trial) < func(best_pos):
                    best_pos = trial

                # Update inertia weight dynamically
                inertia_weight = max(0.4, min(0.9, inertia_weight - 0.001))

            return best_val

        return pso_ade()