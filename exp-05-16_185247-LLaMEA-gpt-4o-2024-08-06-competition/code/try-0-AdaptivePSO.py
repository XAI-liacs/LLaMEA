import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, num_particles=30, inertia_max=0.9, inertia_min=0.4, cognitive=2.0, social=2.0):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_max = inertia_max
        self.inertia_min = inertia_min
        self.cognitive = cognitive
        self.social = social

    def __call__(self, func):
        # Initialize particles
        positions = np.random.uniform(-100, 100, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.num_particles, np.inf)
        
        # Initialize global best
        global_best_score = np.inf
        global_best_position = None
        
        # Optimization loop
        for iteration in range(self.budget // self.num_particles):
            # Evaluate all particles
            for i in range(self.num_particles):
                score = func(positions[i])
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i].copy()

            # Calculate inertia weight
            inertia_weight = self.inertia_max - (self.inertia_max - self.inertia_min) * (iteration / (self.budget // self.num_particles))
            
            # Update velocities and positions
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.cognitive * r1 * (personal_best_positions[i] - positions[i])
                social_component = self.social * r2 * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], -100, 100)
        
        return global_best_score, global_best_position