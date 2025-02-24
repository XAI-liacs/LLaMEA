import numpy as np

class AdaptiveQuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0
        self.c2 = 2.0
        self.alpha = 0.75
        self.inertia = 0.9
        self.inertia_decay = 0.995  # Adjusted inertia decay
        self.mutation_rate = 0.15
        self.particles = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.rand(self.population_size, self.dim) - 0.5
        self.best_particle_positions = np.copy(self.particles)
        self.best_particle_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def __call__(self, func):
        budget_remaining = self.budget
        dynamic_population_size = self.population_size  # Dynamic population size

        while budget_remaining > 0:
            for i in range(dynamic_population_size):
                self.particles[i] = np.clip(self.particles[i], func.bounds.lb, func.bounds.ub)
                
                score = func(self.particles[i])
                budget_remaining -= 1

                if score < self.best_particle_scores[i]:
                    self.best_particle_scores[i] = score
                    self.best_particle_positions[i] = self.particles[i].copy()

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i].copy()
                
                if budget_remaining <= 0:
                    break

            for i in range(dynamic_population_size):
                pb = self.best_particle_positions[i]
                gb = self.global_best_position
                u = np.random.rand(self.dim)
                
                self.velocities[i] = (self.inertia * self.velocities[i] + 
                                      self.c1 * u * (pb - self.particles[i]) + 
                                      self.c2 * (1 - u) * (gb - self.particles[i]))
                self.particles[i] += self.velocities[i] + np.random.normal(scale=0.01, size=self.dim)  # Added perturbation
                
                if np.random.rand() < self.mutation_rate:
                    mutation_vector = np.random.normal(scale=0.1 + (self.global_best_score / self.budget), size=self.dim)
                    self.particles[i] += mutation_vector * (budget_remaining / self.budget)

            self.inertia *= self.inertia_decay
            dynamic_population_size = max(10, dynamic_population_size - 1)  # Adjust population size

        return self.global_best_position, self.global_best_score