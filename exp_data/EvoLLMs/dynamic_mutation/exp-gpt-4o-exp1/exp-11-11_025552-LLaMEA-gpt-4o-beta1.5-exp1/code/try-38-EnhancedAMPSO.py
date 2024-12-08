import numpy as np

class EnhancedAMPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia = 0.7
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.vel_bounds = (-0.5, 0.5)
        self.pos_bounds = (-5.0, 5.0)
        self.memory_rate = 0.1
        self.iterations = self.budget // self.population_size
    
    def oppositional_learning(self, positions):
        opposite_positions = self.pos_bounds[0] + self.pos_bounds[1] - positions
        return opposite_positions

    def __call__(self, func):
        positions = np.random.uniform(self.pos_bounds[0], self.pos_bounds[1], (self.population_size, self.dim))
        velocities = np.random.uniform(self.vel_bounds[0], self.vel_bounds[1], (self.population_size, self.dim))
        
        # Oppositional initialization
        opposite_positions = self.oppositional_learning(positions)
        oppositional_scores = np.array([func(pos) for pos in opposite_positions])
        scores = np.array([func(pos) for pos in positions])
        
        better_initial = oppositional_scores < scores
        positions[better_initial] = opposite_positions[better_initial]
        scores[better_initial] = oppositional_scores[better_initial]

        pbest_positions = positions.copy()
        pbest_scores = scores.copy()
        gbest_position = pbest_positions[np.argmin(pbest_scores)]
        gbest_score = np.min(pbest_scores)

        for i in range(self.iterations):
            self.memory_rate = 0.1 + 0.9 * (i / self.iterations)  # Dynamic adjustment
            inertia_weight = self.inertia * (1 - self.memory_rate * (i / self.iterations))

            # Change 1: Adaptive social coefficient
            self.social_coef = 1.5 + 0.5 * (i / self.iterations)

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (inertia_weight * velocities +
                          self.cognitive_coef * r1 * (pbest_positions - positions) +
                          self.social_coef * r2 * (gbest_position - positions))
            velocities = np.clip(velocities, self.vel_bounds[0], self.vel_bounds[1])

            positions += velocities
            positions = np.clip(positions, self.pos_bounds[0], self.pos_bounds[1])

            scores = np.array([func(pos) for pos in positions])

            better_indices = scores < pbest_scores
            pbest_positions[better_indices] = positions[better_indices]
            pbest_scores[better_indices] = scores[better_indices]

            if np.min(pbest_scores) < gbest_score:
                gbest_score = np.min(pbest_scores)
                gbest_position = pbest_positions[np.argmin(pbest_scores)]

        return gbest_position