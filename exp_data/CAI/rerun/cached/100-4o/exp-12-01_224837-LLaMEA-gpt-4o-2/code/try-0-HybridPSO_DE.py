import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.w = 0.7  # inertia weight
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.F = 0.5  # differential weight
        self.CR = 0.9  # crossover probability
        self.lb = -5.0  # lower bound
        self.ub = 5.0  # upper bound
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.pbest = self.population.copy()
        self.pbest_values = np.full(self.pop_size, np.inf)
        self.gbest = None
        self.gbest_value = np.inf
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            # Evaluate current population
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break
                fitness = func(self.population[i])
                self.evaluations += 1
                if fitness < self.pbest_values[i]:
                    self.pbest_values[i] = fitness
                    self.pbest[i] = self.population[i]
                if fitness < self.gbest_value:
                    self.gbest_value = fitness
                    self.gbest = self.population[i]

            # Update velocities and positions (PSO phase)
            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_vel = self.c1 * r1 * (self.pbest[i] - self.population[i])
                social_vel = self.c2 * r2 * (self.gbest - self.population[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_vel + social_vel
                self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lb, self.ub)

            # Perform DE mutation and crossover (DE phase)
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(self.population[a] + self.F * (self.population[b] - self.population[c]), self.lb, self.ub)
                trial_vector = np.copy(self.population[i])
                jrand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == jrand:
                        trial_vector[j] = mutant[j]

                if self.evaluations < self.budget:
                    trial_fitness = func(trial_vector)
                    self.evaluations += 1
                    if trial_fitness < self.pbest_values[i]:
                        self.pbest_values[i] = trial_fitness
                        self.pbest[i] = trial_vector
                        if trial_fitness < self.gbest_value:
                            self.gbest_value = trial_fitness
                            self.gbest = trial_vector

        return self.gbest