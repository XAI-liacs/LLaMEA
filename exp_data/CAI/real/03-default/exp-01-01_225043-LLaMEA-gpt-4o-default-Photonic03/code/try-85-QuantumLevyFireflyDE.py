import numpy as np

class QuantumLevyFireflyDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 60
        self.individuals = np.random.uniform(size=(self.population_size, dim))
        self.light_intensity = np.full(self.population_size, np.inf)
        self.best_solution = None
        self.best_intensity = np.inf
        self.fitness_evaluations = 0

    def levy_flight(self, L):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=L)
        v = np.random.normal(0, 1, size=L)
        step = u / np.abs(v)**(1 / beta)
        return step

    def move_firefly(self, i, j, alpha, beta, gamma):
        distance = np.linalg.norm(self.individuals[i] - self.individuals[j])
        attractiveness = beta * np.exp(-gamma * distance**2)
        random_factor = alpha * (np.random.rand(self.dim) - 0.5)
        move = attractiveness * (self.individuals[j] - self.individuals[i]) + random_factor
        return move

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        alpha = 0.5
        beta_base = 0.2
        gamma = 1.0

        while self.fitness_evaluations < self.budget:
            for i in range(self.population_size):
                fitness = func(self.individuals[i])
                self.fitness_evaluations += 1
                if fitness < self.light_intensity[i]:
                    self.light_intensity[i] = fitness
                    if fitness < self.best_intensity:
                        self.best_intensity = fitness
                        self.best_solution = self.individuals[i].copy()

            for i in range(self.population_size):
                for j in range(self.population_size):
                    if self.light_intensity[j] < self.light_intensity[i]:
                        move = self.move_firefly(i, j, alpha, beta_base, gamma)
                        self.individuals[i] = np.clip(self.individuals[i] + move, lower_bound, upper_bound)

            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.individuals[indices[0]], self.individuals[indices[1]], self.individuals[indices[2]]

                F = 0.8
                mutant = np.clip(a + F * (b - c), lower_bound, upper_bound)
                crossover_rate = 0.9
                crossover_indices = np.random.rand(self.dim) < crossover_rate
                trial = np.where(crossover_indices, mutant, self.individuals[i])

                trial_fitness = func(trial)
                self.fitness_evaluations += 1

                if trial_fitness < self.light_intensity[i]:
                    self.individuals[i] = trial.copy()
                    self.light_intensity[i] = trial_fitness
                    if trial_fitness < self.best_intensity:
                        self.best_intensity = trial_fitness
                        self.best_solution = trial.copy()

            for i in range(self.population_size):
                levy_step = self.levy_flight(self.dim)
                if np.random.rand() < 0.3:
                    self.individuals[i] += levy_step * (self.individuals[i] - self.best_solution)
                    self.individuals[i] = np.clip(self.individuals[i], lower_bound, upper_bound)

        return self.best_solution