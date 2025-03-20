import numpy as np

class HybridParticleLevyDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = int(budget / dim)
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.velocities = None  # New: Initialize velocities for Particle Swarm Optimization

    def levy_flight(self, beta=1.5):
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / abs(v)**(1 / beta)
        return step

    def particle_swarm_update(self, lb, ub):  # New: Implement Particle Swarm update rule
        w, c1, c2 = 0.5, 1.5, 1.5  # Inertia weight, cognitive and social constants
        for i in range(self.population_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            self.velocities[i] = (w * self.velocities[i] +
                                  c1 * r1 * (self.best_solution - self.population[i]))  # Cognitive component
            self.velocities[i] += c2 * r2 * (self.population[np.random.randint(self.population_size)] - self.population[i])  # Social component
            self.population[i] = np.clip(self.population[i] + self.velocities[i], lb, ub)

    def differential_evolution(self, func, lb, ub):
        F_min, F_max = 0.4, 0.9
        CR = 0.9

        for _ in range(self.budget):
            for i in range(self.population_size):
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                x = self.population[i]
                F = np.random.uniform(F_min, F_max)
                mutant = np.clip(self.population[a] + F * (self.population[b] - self.population[c]), lb, ub)
                
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, x)
                fitness = func(trial)

                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = trial
                
                if fitness < func(x):
                    self.population[i] = trial

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))  # New: Initialize velocities

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        self.differential_evolution(func, lb, ub)

        for _ in range(self.budget // self.population_size):
            self.particle_swarm_update(lb, ub)  # New: Use Particle Swarm update
            for i in range(self.population_size):
                step = self.levy_flight()
                candidate = np.clip(self.population[i] + step, lb, ub)
                fitness = func(candidate)
                
                if fitness < func(self.population[i]):
                    self.population[i] = candidate

                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = candidate

        return self.best_solution