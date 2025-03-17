import numpy as np

class AntSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, self.budget // 10)  # Ensure enough evaluations
        self.ants = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        self.personal_best_positions = self.ants.copy()
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def __call__(self, func):
        if not hasattr(func, 'bounds') or not hasattr(func.bounds, 'lb') or not hasattr(func.bounds, 'ub'):
            raise ValueError("Function must have 'bounds' with 'lb' and 'ub' attributes")

        lb, ub = func.bounds.lb, func.bounds.ub

        evaluations = 0
        while evaluations < self.budget:
            # Evaluate current positions
            scores = np.array([func(ant) for ant in self.ants])
            evaluations += len(self.ants)

            # Update personal bests
            better_mask = scores < self.personal_best_scores
            self.personal_best_positions[better_mask] = self.ants[better_mask]
            self.personal_best_scores[better_mask] = scores[better_mask]

            # Update global best
            min_score_idx = np.argmin(scores)
            if scores[min_score_idx] < self.global_best_score:
                self.global_best_score = scores[min_score_idx]
                self.global_best_position = self.ants[min_score_idx]

            # Update velocities and positions
            inertia = 0.7
            cognitive = 1.5
            social = 1.5
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            self.velocities = (inertia * self.velocities +
                               cognitive * r1 * (self.personal_best_positions - self.ants) +
                               social * r2 * (self.global_best_position - self.ants))
            self.ants += self.velocities

            # Clip positions to bounds
            self.ants = np.clip(self.ants, lb, ub)

        return self.global_best_position, self.global_best_score