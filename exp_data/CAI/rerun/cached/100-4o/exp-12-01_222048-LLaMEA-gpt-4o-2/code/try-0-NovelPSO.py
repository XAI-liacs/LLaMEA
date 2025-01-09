import numpy as np

class NovelPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 50
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.inertia_max = 0.9
        self.inertia_min = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.vel_max = (self.upper_bound - self.lower_bound) / 2.0

    def __call__(self, func):
        np.random.seed(0)  # For reproducibility
        
        # Initialize particle positions and velocities
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.random.uniform(-self.vel_max, self.vel_max, (self.num_particles, self.dim))
        
        # Initialize personal best and global best
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, np.inf)
        global_best_position = None
        global_best_score = np.inf
        
        n_evaluations = 0
        
        while n_evaluations < self.budget:
            # Evaluate each particle's fitness
            for i in range(self.num_particles):
                score = func(positions[i])
                n_evaluations += 1
                
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

                if n_evaluations >= self.budget:
                    break
            
            # Dynamic adjustment of inertia weight
            inertia = self.inertia_max - ((self.inertia_max - self.inertia_min) * (n_evaluations / self.budget))
            
            # Update velocities and positions
            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                cognitive_velocity = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = self.c2 * r2 * (global_best_position - positions[i])

                velocities[i] = inertia * velocities[i] + cognitive_velocity + social_velocity
                
                # Clamp velocity
                velocities[i] = np.clip(velocities[i], -self.vel_max, self.vel_max)

                # Update position
                positions[i] += velocities[i]
                
                # Ensure positions are within bounds
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

        return global_best_position, global_best_score