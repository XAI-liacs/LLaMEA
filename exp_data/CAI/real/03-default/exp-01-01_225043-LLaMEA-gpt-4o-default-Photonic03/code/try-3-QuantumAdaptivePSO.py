import numpy as np

class QuantumAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.rand(self.population_size, dim) * 0.1
        self.personal_best = self.particles.copy()
        self.global_best = None
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best_fitness = np.inf
        self.fitness_evaluations = 0
        self.inertia_max = 0.9
        self.inertia_min = 0.4
        self.quantum_delta = 0.1

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]
        while self.fitness_evaluations < self.budget:
            # Non-linear decrease of inertia over time
            inertia_weight = self.inertia_max - (self.inertia_max - self.inertia_min) * (self.fitness_evaluations / self.budget)**2
            
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                # Evaluate particle fitness
                fitness = func(self.particles[i])
                self.fitness_evaluations += 1

                # Update personal and global bests
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.particles[i].copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.particles[i].copy()

            # Update particles using Quantum-Inspired PSO
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                quantum_potential = (self.global_best - self.particles[i]) * self.quantum_delta * np.random.standard_normal(self.dim)
                cognitive_coeff = 1.5 + 0.5 * np.random.rand()
                social_coeff = 1.5 + 0.5 * np.random.rand()
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_velocity = cognitive_coeff * r1 * (self.personal_best[i] - self.particles[i])
                social_velocity = social_coeff * r2 * (self.global_best - self.particles[i])
                
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity + quantum_potential
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lower_bound, upper_bound)

        return self.global_best