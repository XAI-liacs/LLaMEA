import numpy as np

class AdaptivePSOWithDynamicPopulation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_initial = 2.5
        self.c2_initial = 1.5
        self.c1_final = 1.5
        self.c2_final = 2.5
        self.velocity_clamp = 0.5
        self.initial_population_size = 20
        self.max_population_size = 40
    
    def __call__(self, func):
        np.random.seed(0)
        num_particles = self.initial_population_size
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(num_particles, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        evaluations = 0
        while evaluations < self.budget:
            for i in range(num_particles):
                score = func(positions[i])
                evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = positions[i].copy()
                
                if evaluations >= self.budget:
                    break
            
            inertia_weight = self.w_max - ((self.w_max - self.w_min) * evaluations / self.budget)
            c1 = self.c1_initial - ((self.c1_initial - self.c1_final) * evaluations / self.budget)
            c2 = self.c2_initial + ((self.c2_final - self.c2_initial) * evaluations / self.budget)
            
            for i in range(num_particles):
                cognitive_component = c1 * np.random.uniform(0, 1, self.dim) * (personal_best_positions[i] - positions[i])
                social_component = c2 * np.random.uniform(0, 1, self.dim) * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                
                velocities[i] = np.clip(velocities[i], -self.velocity_clamp, self.velocity_clamp)
                
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

            # Adaptively adjust the population size
            if evaluations < self.budget // 2:
                num_particles = min(num_particles + 1, self.max_population_size)
                if num_particles > positions.shape[0]:
                    additional_positions = np.random.uniform(self.lower_bound, self.upper_bound, (1, self.dim))
                    additional_velocities = np.random.uniform(-1, 1, (1, self.dim))
                    positions = np.vstack((positions, additional_positions))
                    velocities = np.vstack((velocities, additional_velocities))
                    personal_best_positions = np.vstack((personal_best_positions, additional_positions))
                    personal_best_scores = np.append(personal_best_scores, float('inf'))