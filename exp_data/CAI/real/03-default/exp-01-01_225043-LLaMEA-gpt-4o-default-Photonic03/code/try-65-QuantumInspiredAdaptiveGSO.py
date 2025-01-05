import numpy as np

class QuantumInspiredAdaptiveGSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.uniform(size=(self.population_size, dim))
        self.velocities = np.random.uniform(size=(self.population_size, dim)) * 0.1
        self.personal_best = self.particles.copy()
        self.global_best = None
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best_fitness = np.inf
        self.fitness_evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        while self.fitness_evaluations < self.budget:
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                # Evaluate fitness & update personal and global bests
                fitness = func(self.particles[i])
                self.fitness_evaluations += 1
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.particles[i].copy()
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.particles[i].copy()

            # Quantum potential wells exploration
            quantum_well_width = 0.1 * (upper_bound - lower_bound)
            for i in range(self.population_size):
                if np.random.rand() < 0.1:  # Quantum jump probability
                    quantum_jump = lower_bound + np.random.rand(self.dim) * quantum_well_width
                    self.particles[i] = quantum_jump

            # Adaptive velocity update (Quantum-inspired)
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break
                inertia_weight = 0.5 + 0.4 * np.sin(2 * np.pi * self.fitness_evaluations / self.budget)
                cognitive_coeff = 2.0
                social_coeff = 2.0
                r1, r2 = np.random.rand(), np.random.rand()

                cognitive_velocity = cognitive_coeff * r1 * (self.personal_best[i] - self.particles[i])
                social_velocity = social_coeff * r2 * (self.global_best - self.particles[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lower_bound, upper_bound)

            # Genetic-like crossover for diversity enhancement
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break
                partner_idx = np.random.randint(0, self.population_size)
                crossover_prob = 0.6
                mask = np.random.rand(self.dim) < crossover_prob
                offspring = np.where(mask, self.particles[i], self.particles[partner_idx])
                
                offspring_fitness = func(offspring)
                self.fitness_evaluations += 1

                if offspring_fitness < self.personal_best_fitness[i]:
                    self.particles[i] = offspring.copy()
                    self.personal_best[i] = offspring.copy()
                    self.personal_best_fitness[i] = offspring_fitness
                    if offspring_fitness < self.global_best_fitness:
                        self.global_best_fitness = offspring_fitness
                        self.global_best = offspring.copy()

        return self.global_best