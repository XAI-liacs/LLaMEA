import numpy as np

class QuantumSwarmSimulatedAnnealing:
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
        self.temperature = 1.0  # Initial temperature for simulated annealing

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]
        
        while self.fitness_evaluations < self.budget:
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                fitness = func(self.particles[i])
                self.fitness_evaluations += 1

                # Update personal and global bests
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.particles[i].copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.particles[i].copy()

            # Simulated annealing acceptance criterion
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                inertia_weight = 0.9 - 0.7 * (self.fitness_evaluations / self.budget)
                cognitive_coeff = 1.5
                social_coeff = 1.5
                r1, r2 = np.random.rand(), np.random.rand()

                # Velocity update
                cognitive_velocity = cognitive_coeff * r1 * (self.personal_best[i] - self.particles[i])
                social_velocity = social_coeff * r2 * (self.global_best - self.particles[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lower_bound, upper_bound)

                # Quantum-inspired perturbation
                if np.random.rand() < np.exp(-self.fitness_evaluations / self.budget):
                    q_jump = np.random.uniform(-0.1, 0.1, self.dim)
                    self.particles[i] += q_jump
                    self.particles[i] = np.clip(self.particles[i], lower_bound, upper_bound)

                # Evaluate new position
                trial_fitness = func(self.particles[i])
                self.fitness_evaluations += 1

                # Accept trial solution based on simulated annealing criterion
                if trial_fitness < self.personal_best_fitness[i] or \
                   np.random.rand() < np.exp((self.personal_best_fitness[i] - trial_fitness) / self.temperature):
                    self.personal_best[i] = self.particles[i].copy()
                    self.personal_best_fitness[i] = trial_fitness

                    # Update global best if necessary
                    if trial_fitness < self.global_best_fitness:
                        self.global_best_fitness = trial_fitness
                        self.global_best = self.particles[i].copy()
            
            # Decrease temperature
            self.temperature *= 0.99

        return self.global_best