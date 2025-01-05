import numpy as np

class QuantumLevySwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 60
        self.particles = np.random.uniform(size=(self.swarm_size, dim))
        self.velocities = np.random.uniform(-0.1, 0.1, size=(self.swarm_size, dim))
        self.personal_best_positions = self.particles.copy()
        self.global_best_position = None
        self.personal_best_fitness = np.full(self.swarm_size, np.inf)
        self.global_best_fitness = np.inf
        self.fitness_evaluations = 0

    def levy_flight(self, L):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=L)
        v = np.random.normal(0, 1, size=L)
        step = u / np.abs(v)**(1 / beta)
        return step

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        def inertia_weight(evals):
            return 0.9 - 0.5 * (evals / self.budget)

        def cognitive_coeff(evals):
            return 2.5 - 1.5 * (evals / self.budget)

        def social_coeff(evals):
            return 0.5 + 1.5 * (evals / self.budget)

        while self.fitness_evaluations < self.budget:
            for i in range(self.swarm_size):
                if self.fitness_evaluations >= self.budget:
                    break

                fitness = func(self.particles[i])
                self.fitness_evaluations += 1

                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.particles[i].copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = self.particles[i].copy()

            for i in range(self.swarm_size):
                if self.fitness_evaluations >= self.budget:
                    break

                w = inertia_weight(self.fitness_evaluations)
                c1 = cognitive_coeff(self.fitness_evaluations)
                c2 = social_coeff(self.fitness_evaluations)

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                cognitive_component = c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_component = c2 * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = w * self.velocities[i] + cognitive_component + social_component

                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lower_bound, upper_bound)

                levy_step = self.levy_flight(self.dim)
                if np.random.rand() < 0.3:
                    self.particles[i] += levy_step * (self.particles[i] - self.global_best_position)
                    self.particles[i] = np.clip(self.particles[i], lower_bound, upper_bound)

        return self.global_best_position