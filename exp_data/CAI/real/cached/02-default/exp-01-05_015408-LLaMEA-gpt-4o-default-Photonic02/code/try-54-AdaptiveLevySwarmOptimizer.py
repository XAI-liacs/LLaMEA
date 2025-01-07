import numpy as np

class AdaptiveLevySwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 1.5  # Levy exponent
        self.beta = 0.5   # Direct movement influence
        self.adaptive_factor = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        velocities = np.random.randn(self.population_size, self.dim) * 0.1
        fitness = np.array([func(ind) for ind in swarm])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = swarm[best_index]

        while evaluations < self.budget:
            for i in range(self.population_size):
                levy_jump = self.levy_flight(self.dim, self.alpha) * (swarm[best_index] - swarm[i])
                velocities[i] = self.beta * velocities[i] + levy_jump
                swarm[i] = swarm[i] + velocities[i]

                # Ensure particles are within bounds
                swarm[i] = np.clip(swarm[i], lb, ub)

                # Evaluate new position
                new_fitness = func(swarm[i])
                evaluations += 1

                # Update fitness and best position
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness

                if new_fitness < fitness[best_index]:
                    best_index = i
                    best_position = swarm[i]

                if evaluations >= self.budget:
                    break

        return best_position, fitness[best_index]

    def levy_flight(self, dim, alpha):
        # Generate a Levy flight step
        sigma_u = (np.gamma(1 + alpha) * np.sin(np.pi * alpha / 2) / 
                   (np.gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))) ** (1 / alpha)
        u = np.random.randn(dim) * sigma_u
        v = np.random.randn(dim)
        step = u / np.abs(v) ** (1 / alpha)
        return step