import numpy as np

class FireflyLevyCauchy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.5
        self.beta_0 = 1.0
        self.gamma = 0.1
        self.step_size = 0.1

    def levy_flight(self):
        sigma = (np.math.gamma(1 + self.beta_0) * np.math.sin(np.pi * self.beta_0 / 2) / (np.math.gamma((1 + self.beta_0) / 2) * self.beta_0 * 2 ** ((self.beta_0 - 1) / 2))) ** (1 / self.beta_0)
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        step = u / abs(v) ** (1 / self.beta_0)
        return step

    def cauchy_mutation(self, x):
        mutated_x = x + self.gamma / (1 + self.step_size) * np.tan(np.pi * (np.random.rand() - 0.5))
        return mutated_x

    def __call__(self, func):
        population = 10 * np.random.rand(self.population_size, self.dim) - 5.0
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[j] < fitness[i]:
                        attractiveness = self.alpha * np.exp(-np.linalg.norm(population[j] - population[i])**2)
                        step = self.levy_flight()
                        new_position = population[i] + attractiveness * (population[j] - population[i]) + step
                        new_position = np.clip(new_position, -5.0, 5.0)
                        new_position = self.cauchy_mutation(new_position)
                        new_fitness = func(new_position)
                        
                        if new_fitness < fitness[i]:
                            population[i] = new_position
                            fitness[i] = new_fitness

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        return best_solution