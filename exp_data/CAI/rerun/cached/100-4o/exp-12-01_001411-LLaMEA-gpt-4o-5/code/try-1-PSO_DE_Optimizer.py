import numpy as np

class PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-abs(self.upper_bound - self.lower_bound), abs(self.upper_bound - self.lower_bound), (self.population_size, self.dim))
        self.pbest_positions = np.copy(self.particles)
        self.pbest_scores = np.full(self.population_size, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf
        self.cr = 0.9  # Crossover rate for DE
        self.f = 0.8   # Differential weight for DE

    def __call__(self, func):
        evaluations = 0
        
        def update_velocities_and_positions():
            nonlocal evaluations
            inertia_weight = 0.5 + np.random.rand() / 2
            cognitive_component = 2.05 * np.random.rand(self.population_size, self.dim)
            social_component = 2.05 * np.random.rand(self.population_size, self.dim)

            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_velocity = cognitive_component[i] * (self.pbest_positions[i] - self.particles[i])
                social_velocity = social_component[i] * (self.gbest_position - self.particles[i] if self.gbest_position is not None else 0)
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
                self.particles[i] += self.velocities[i]

                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)
                score = func(self.particles[i])
                evaluations += 1

                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.particles[i]

                if self.gbest_position is None or score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = self.particles[i]
        
        def differential_evolution():
            nonlocal evaluations
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = self.particles[a] + self.f * (self.particles[b] - self.particles[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                trial_vector = np.copy(self.particles[i])
                crossover_points = np.random.rand(self.dim) < self.cr
                if not np.any(crossover_points):
                    crossover_points[np.random.randint(0, self.dim)] = True
                trial_vector[crossover_points] = mutant_vector[crossover_points]

                score = func(trial_vector)
                evaluations += 1

                if score < func(self.particles[i]):
                    self.particles[i] = trial_vector
                    if score < self.pbest_scores[i]:
                        self.pbest_scores[i] = score
                        self.pbest_positions[i] = trial_vector
                    if score < self.gbest_score:
                        self.gbest_score = score
                        self.gbest_position = trial_vector

        while evaluations < self.budget:
            update_velocities_and_positions()
            differential_evolution()

        return self.gbest_position, self.gbest_score