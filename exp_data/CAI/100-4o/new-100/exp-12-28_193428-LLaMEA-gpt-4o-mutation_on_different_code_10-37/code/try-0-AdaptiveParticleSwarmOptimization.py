import numpy as np

class AdaptiveParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, population_size=30):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None
        
        self.w_max = 0.9  # maximum inertia weight
        self.w_min = 0.4  # minimum inertia weight
        self.c1 = 2.0  # cognitive component
        self.c2 = 2.0  # social component

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub

        # Initialize positions and velocities
        positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, np.inf)
        
        # Evaluate initial population
        for i in range(self.population_size):
            score = func(positions[i])
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i]
            if score < self.f_opt:
                self.f_opt = score
                self.x_opt = positions[i]

        num_evaluations = self.population_size

        while num_evaluations < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * (num_evaluations / self.budget)
            
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (w * velocities[i] 
                                 + self.c1 * r1 * (personal_best_positions[i] - positions[i]) 
                                 + self.c2 * r2 * (self.x_opt - positions[i]))

                positions[i] += velocities[i]
                # Ensure particle remains within bounds
                positions[i] = np.clip(positions[i], lb, ub)

                score = func(positions[i])
                num_evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = positions[i]

                if num_evaluations >= self.budget:
                    break
            
        return self.f_opt, self.x_opt