import numpy as np

class HybridPSOwithADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.F = 0.8  # DE mutation factor
        self.CR = 0.9  # DE crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.best_position = None
        self.best_value = np.inf
        
    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.array([func(p) for p in particles])
        evaluations = self.population_size
        
        while evaluations < self.budget:
            # Update PSO velocities and positions
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coefficient * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.social_coefficient * r2 * (self.best_position - particles[i]))
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

            # Differential Evolution step
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = particles[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                trial_vector = np.array([mutant_vector[j] if np.random.rand() < self.CR else particles[i, j] for j in range(self.dim)])
                trial_value = func(trial_vector)
                evaluations += 1
                if trial_value < personal_best_values[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_values[i] = trial_value
                if trial_value < self.best_value:
                    self.best_value = trial_value
                    self.best_position = trial_vector

            # Update global best
            for i in range(self.population_size):
                value = func(particles[i])
                evaluations += 1
                if value < personal_best_values[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_values[i] = value
                if value < self.best_value:
                    self.best_value = value
                    self.best_position = particles[i]

        return self.best_position, self.best_value