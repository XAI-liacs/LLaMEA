import numpy as np

class AdaptiveQuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0
        self.c2 = 2.0
        self.alpha = 0.75  # Constriction factor
        self.inertia = 0.9  # Dynamic inertia
        self.mutation_rate = 0.1  # Mutation rate
        self.particles = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.rand(self.population_size, self.dim) - 0.5
        self.best_particle_positions = np.copy(self.particles)
        self.best_particle_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def __call__(self, func):
        budget_remaining = self.budget
        while budget_remaining > 0:
            for i in range(self.population_size):
                # Ensure particles are within bounds
                self.particles[i] = np.clip(self.particles[i], func.bounds.lb, func.bounds.ub)
                
                # Evaluate particle
                score = func(self.particles[i])
                budget_remaining -= 1

                # Update personal best
                if score < self.best_particle_scores[i]:
                    self.best_particle_scores[i] = score
                    self.best_particle_positions[i] = self.particles[i].copy()

                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i].copy()
                
                if budget_remaining <= 0:
                    break

            # Update particles using quantum behavior
            for i in range(self.population_size):
                # Quantum behavior - simulate the motion of particles in a potential well
                pb = self.best_particle_positions[i]
                gb = self.global_best_position
                u = np.random.rand(self.dim)
                
                # Adaptive velocity update with dynamic inertia
                self.velocities[i] = (self.inertia * np.random.rand() * self.velocities[i] +  # Line modified
                                      self.alpha * (u * (pb - self.particles[i]) + (1 - u) * (gb - self.particles[i])))
                self.particles[i] += self.velocities[i]
                
                # Mutation for enhanced exploration
                if np.random.rand() < self.mutation_rate * 0.5:  # Line modified
                    self.particles[i] += np.random.normal(size=self.dim) * 0.1 * np.random.rand()  # Line modified

        return self.global_best_position, self.global_best_score