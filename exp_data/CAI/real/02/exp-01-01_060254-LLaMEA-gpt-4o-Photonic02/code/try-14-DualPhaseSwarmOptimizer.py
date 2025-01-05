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

        # Adaptive factor based on dynamic exploration feedback
        adaptive_factor = 0.15 + 0.4 * np.exp(-best_value / (best_value + 1e-9))  # Changed line

        # Phase 2: Exploitation with local search
        remaining_budget = self.budget - samples
        local_best_position = np.copy(best_position)
        local_best_value = best_value
        
        for _ in range(remaining_budget):
            for d in range(self.dim):
                step = adaptive_factor * (ub[d] - lb[d]) * (np.random.rand() - 0.5)
                candidate_position = np.copy(local_best_position)
                candidate_position[d] += step * np.random.uniform(0.4, 1.6)  # Changed line

                # Ensure the candidate stays within bounds
                candidate_position = np.clip(candidate_position, lb, ub)
                
                candidate_value = func(candidate_position)
                if candidate_value < local_best_value:
                    local_best_value = candidate_value
                    local_best_position = candidate_position

        return local_best_position, local_best_value