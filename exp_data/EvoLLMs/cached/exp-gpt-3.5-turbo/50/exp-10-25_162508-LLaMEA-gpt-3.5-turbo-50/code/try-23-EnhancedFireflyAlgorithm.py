import numpy as np

class EnhancedFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20 * dim
        self.alpha = 0.2
        self.beta_min = 0.2
        self.gamma = 1.0
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.levy_mu = 1.5  # Levy flight exponent
        self.levy_scale = 0.1  # Levy flight scale factor

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

    def _get_fitness(self, population, func):
        return np.array([func(individual) for individual in population])

    def _attractiveness(self, i, j):
        return self.beta_min + (self.alpha - self.beta_min) * np.exp(-self.gamma * np.linalg.norm(i - j))

    def _update_position(self, individual, best_individual):
        levy = np.random.standard_cauchy(self.dim) / (np.power(np.abs(np.random.normal(0, 1, self.dim)), 1 / self.levy_mu))
        new_position = individual + self._attractiveness(best_individual, individual) * (best_individual - individual) + self.levy_scale * levy
        return np.clip(new_position, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        population = self._initialize_population()
        evals = 0

        while evals < self.budget:
            fitness_values = self._get_fitness(population, func)
            best_individual = population[np.argmin(fitness_values)]

            for i in range(self.pop_size):
                new_position = self._update_position(population[i], best_individual)
                population[i] = new_position
                evals += 1

                if evals >= self.budget:
                    break

        best_solution = population[np.argmin(self._get_fitness(population, func))]
        return best_solution