import numpy as np

class Enhanced_Hybrid_PSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Enhanced Particle Swarm Optimization parameters
        self.num_particles = 50  # Increased particle count for diversity
        self.inertia_weight = 0.9  # Higher inertia weight for global exploration
        self.cognitive_coeff = 1.4
        self.social_coeff = 1.6  # Slightly higher social impact for convergence

        # Adaptive Differential Evolution parameters
        self.base_F = 0.7  # Base scaling factor for adaptability
        self.base_CR = 0.8  # Base crossover probability

        # Initialize particles and velocities
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-0.6, 0.6, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def chaotic_map(self, x):
        return 3.9 * x * (1 - x)  # Adjusted chaotic map

    def levy_flight(self, L):
        beta = 1.7  # Adjusted beta for Lévy stability
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, L)
        v = np.random.normal(0, 1, L)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def __call__(self, func):
        evals = 0
        chaos_factor = np.random.rand()
        
        while evals < self.budget:
            # Evaluate each particle
            scores = np.apply_along_axis(func, 1, self.positions)
            evals += self.num_particles

            # Update personal and global bests
            for i in range(self.num_particles):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i]

                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i]

            # Update velocities and positions (Enhanced PSO)
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            self.velocities = (self.inertia_weight * self.velocities + cognitive_component + social_component) * chaos_factor
            self.positions += self.velocities * np.random.uniform(0.2, 0.6, self.positions.shape)  # Slightly broader range
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Perform Adaptive Differential Evolution with Dynamic Lévy flights
            for i in range(self.num_particles):
                F = self.base_F + (0.2 * np.random.rand())  # Adaptive F
                CR = self.base_CR + (0.1 * np.random.rand())  # Adaptive CR

                indices = [idx for idx in range(self.num_particles) if idx != i]
                x1, x2, x3 = self.positions[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(x1 + F * (x2 - x3), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < CR, mutant_vector, self.positions[i])
                
                # Incorporate Dynamic Levy flights
                levy_steps = self.levy_flight(self.dim)
                trial_vector += 0.02 * levy_steps * (trial_vector - self.positions[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                
                trial_score = func(trial_vector)

                # DE acceptance criterion
                if trial_score < scores[i]:
                    self.positions[i] = trial_vector
                    scores[i] = trial_score
            
            chaos_factor = self.chaotic_map(chaos_factor)  # Update chaos factor for next iteration
            evals += self.num_particles

        return self.global_best_position, self.global_best_score