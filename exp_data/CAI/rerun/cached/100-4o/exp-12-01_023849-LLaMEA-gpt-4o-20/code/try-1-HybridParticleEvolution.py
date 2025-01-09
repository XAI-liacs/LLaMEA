import numpy as np

class HybridParticleEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.inertia_weight = 0.7
        self.cognitive_coef = 1.5
        self.social_coef = 1.5

    def __call__(self, func):
        np.random.seed(42)
        # Initialize particles
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_values)]
        
        # Initialize evolutionary strategy
        generation = 0
        evaluations = self.population_size

        while evaluations < self.budget:
            new_particles = np.copy(particles)
            for i in range(self.population_size):
                # DE mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = particles[indices]
                mutant_vector = np.clip(x1 + self.mutation_factor * (x2 - x3), self.lower_bound, self.upper_bound)
                
                # DE crossover
                trial_vector = np.copy(particles[i])
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_rate:
                        trial_vector[j] = mutant_vector[j]

                # PSO update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                self.cognitive_coef * r1 * (personal_best_positions[i] - particles[i]) +
                                self.social_coef * r2 * (global_best_position - particles[i]))
                new_particles[i] = np.clip(trial_vector + velocities[i], self.lower_bound, self.upper_bound)

            # Evaluate new particles
            new_particle_values = np.array([func(p) for p in new_particles])
            evaluations += self.population_size

            # Update personal and global bests
            for i in range(self.population_size):
                if new_particle_values[i] < personal_best_values[i]:
                    personal_best_values[i] = new_particle_values[i]
                    personal_best_positions[i] = new_particles[i]
            if np.min(new_particle_values) < np.min(personal_best_values):
                global_best_position = new_particles[np.argmin(new_particle_values)]

            # Update particles
            particles = new_particles
            generation += 1

        return global_best_position