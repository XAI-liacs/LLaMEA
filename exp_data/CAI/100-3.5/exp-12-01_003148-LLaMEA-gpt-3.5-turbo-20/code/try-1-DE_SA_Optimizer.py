import numpy as np

class DE_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.initial_mutation_factor = 0.5
        self.final_mutation_factor = 0.1
        self.crossover_prob = 0.7
        self.initial_temperature = 1.0
        self.final_temperature = 0.1
        self.alpha = 0.99

    def __call__(self, func):
        def random_vector(bounds):
            return bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(self.dim)

        def clipToBounds(x, bounds):
            return np.clip(x, bounds[0], bounds[1])

        def mutate(x, population, bounds, mutation_factor):
            indices = np.random.choice(len(population), 3, replace=False)
            a, b, c = population[indices]
            mutant = clipToBounds(a + mutation_factor * (b - c), bounds)
            return mutant

        def crossover(x, mutant, bounds):
            trial = np.copy(x)
            j = np.random.randint(self.dim)
            for indx in range(self.dim):
                if np.random.rand() > self.crossover_prob and indx != j:
                    trial[indx] = mutant[indx]
            return clipToBounds(trial, bounds)

        def adjust_mutation_factor(iteration):
            return self.initial_mutation_factor - (self.initial_mutation_factor - self.final_mutation_factor) * iteration / self.budget

        def accept(candidate, x, temperature):
            if candidate < x:
                return candidate
            elif np.random.uniform() < np.exp((x - candidate) / temperature):
                return candidate
            return x

        def anneal(x, population, bounds, temperature, mutation_factor):
            candidate = mutate(x, population, bounds, mutation_factor)
            trial = crossover(x, candidate, bounds)
            return accept(func(trial), x, temperature)

        bounds = (-5.0, 5.0)
        population = [random_vector(bounds) for _ in range(self.population_size)]
        x = random_vector(bounds)
        temperature = self.initial_temperature

        for i in range(self.budget):
            mutation_factor = adjust_mutation_factor(i)
            x = anneal(x, population, bounds, temperature, mutation_factor)
            temperature *= self.alpha
        return x