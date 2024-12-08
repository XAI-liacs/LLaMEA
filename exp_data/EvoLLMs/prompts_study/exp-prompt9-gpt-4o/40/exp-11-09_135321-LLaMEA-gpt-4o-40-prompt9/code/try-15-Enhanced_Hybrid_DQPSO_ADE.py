import numpy as np

class Enhanced_Hybrid_DQPSO_ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # DQPSO-ADE parameters
        self.num_particles = 50  # Increased number of particles for diverse exploration
        self.inertia_weight_bounds = (0.9, 0.4)  # Adaptive inertia weight
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.7

        # Differential Evolution parameters
        self.F_bounds = (0.5, 0.9)  # Adaptive F for flexibility
        self.CR = 0.9

        # Initialize particles in quantum space
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def levy_flight(self, L):
        return np.random.standard_cauchy(size=L)

    def adaptive_inertia_weight(self, evals):
        alpha = evals / self.budget
        return self.inertia_weight_bounds[1] + (self.inertia_weight_bounds[0] - self.inertia_weight_bounds[1]) * (1 - alpha)

    def adaptive_F(self, evals):
        alpha = evals / self.budget
        return self.F_bounds[0] + (self.F_bounds[1] - self.F_bounds[0]) * alpha

    def __call__(self, func):
        evals = 0
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

            # Update velocities and positions (DQPSO)
            inertia_weight = self.adaptive_inertia_weight(evals)
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            self.velocities = inertia_weight * self.velocities + cognitive_component + social_component
            self.positions += self.velocities * np.random.uniform(0.1, 0.5, self.positions.shape)
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Perform Differential Evolution with Lévy flights and FDRS
            F = self.adaptive_F(evals)
            for i in range(self.num_particles):
                indices = [idx for idx in range(self.num_particles) if idx != i]
                x1, x2, x3 = self.positions[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(x1 + F * (x2 - x3), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.CR, mutant_vector, self.positions[i])
                
                # Incorporate Levy flights for better exploration
                levy_steps = self.levy_flight(self.dim)
                trial_vector += 0.01 * levy_steps * (trial_vector - self.positions[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                
                trial_score = func(trial_vector)

                # DE acceptance criterion
                if trial_score < scores[i]:
                    self.positions[i] = trial_vector
                    scores[i] = trial_score
                    
            evals += self.num_particles

        return self.global_best_position, self.global_best_score