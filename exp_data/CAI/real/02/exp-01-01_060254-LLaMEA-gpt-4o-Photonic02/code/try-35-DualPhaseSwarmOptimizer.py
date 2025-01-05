import numpy as np

class DualPhaseSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_position = None
        best_value = float('inf')
        
        # Phase 1: Exploration with random sampling
        samples = int(self.budget * 0.6)
        for _ in range(samples):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            value = func(position)
            if value < best_value:
                best_value = value
                best_position = position

        # Enhanced adaptive factor with nonlinear decay
        adaptive_factor = 0.1 + 0.4 * np.exp(-np.sqrt(best_value / (best_value + 1e-9)))

        # Phase 2: Exploitation with local search
        remaining_budget = self.budget - samples
        local_best_position = np.copy(best_position)
        local_best_value = best_value
        
        for i in range(remaining_budget):
            decay = 1 - (i / remaining_budget)  
            inertia_weight = 0.9 - 0.5 * adaptive_factor * (i / remaining_budget) + 0.1 * np.sin(np.pi * i / remaining_budget)  # Changed line: introduce adaptive sinusoidal dynamic inertia weight
            for d in range(self.dim):
                step_scaling = np.random.rand()
                step = inertia_weight * adaptive_factor * (ub[d] - lb[d]) * (np.random.rand() - 0.5) * decay * step_scaling
                candidate_position = np.copy(local_best_position)
                candidate_position[d] += step * np.random.uniform(0.5, 1.5)

                # Ensure the candidate stays within bounds
                candidate_position = np.clip(candidate_position, lb, ub)
                
                candidate_value = func(candidate_position)
                if candidate_value < local_best_value:
                    local_best_value = candidate_value
                    local_best_position = candidate_position

        return local_best_position, local_best_value