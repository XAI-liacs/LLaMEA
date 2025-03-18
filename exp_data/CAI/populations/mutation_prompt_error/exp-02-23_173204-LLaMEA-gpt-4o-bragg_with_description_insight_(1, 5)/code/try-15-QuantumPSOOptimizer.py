import numpy as np
from scipy.optimize import minimize

class QuantumPSOOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.velocity = np.zeros((self.population_size, self.dim))
        self.best_personal_positions = None
        self.best_personal_scores = np.full(self.population_size, float('inf'))
        self.best_global_position = None
        self.best_global_score = float('inf')
        self.c1 = 2.0  # Cognitive constant
        self.c2 = 2.0  # Social constant
        self.w = 0.5  # Inertia weight

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def periodicity_enforcement(self, solution, period=2):
        return np.repeat(np.mean(solution.reshape(-1, period), axis=1), period)

    def adaptive_topology(self, generation):
        # Adapt inertia weight and social/cognitive constants based on generation
        self.w = 0.9 - 0.5 * (generation / (self.budget // self.population_size))
        self.c1 = 1.5 + 0.5 * np.sin(np.pi * generation / (self.budget // self.population_size))
        self.c2 = 1.5 + 0.5 * np.cos(np.pi * generation / (self.budget // self.population_size))

    def quantum_particle_swarm(self, func, bounds):
        lb, ub = bounds.lb, bounds.ub
        population = self.initialize_population(lb, ub)

        for generation in range(self.budget // self.population_size):
            self.adaptive_topology(generation)

            for i in range(self.population_size):
                score = func(population[i])
                if score < self.best_personal_scores[i]:
                    self.best_personal_scores[i] = score
                    self.best_personal_positions[i] = population[i]
                if score < self.best_global_score:
                    self.best_global_score = score
                    self.best_global_position = population[i]

                # Update velocity and position
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.best_personal_positions[i] - population[i])
                social_component = self.c2 * r2 * (self.best_global_position - population[i])
                self.velocity[i] = self.w * self.velocity[i] + cognitive_component + social_component

                # Apply velocity and enforce periodicity
                population[i] = population[i] + self.velocity[i]
                population[i] = self.periodicity_enforcement(population[i])
                population[i] = np.clip(population[i], lb, ub)

            # Perform local optimization occasionally
            if generation % 10 == 0 and self.best_global_position is not None:
                self.local_optimization(func, self.best_global_position, bounds)

    def local_optimization(self, func, initial_solution, bounds):
        result = minimize(func, initial_solution, bounds=list(zip(bounds.lb, bounds.ub)), method='L-BFGS-B')
        if result.fun < self.best_global_score:
            self.best_global_score = result.fun
            self.best_global_position = result.x

    def __call__(self, func):
        bounds = func.bounds
        self.best_personal_positions = self.initialize_population(bounds.lb, bounds.ub)
        self.quantum_particle_swarm(func, bounds)
        
        # Perform final local optimization on the best found solution
        if self.best_global_position is not None:
            self.local_optimization(func, self.best_global_position, bounds)
        
        return self.best_global_position