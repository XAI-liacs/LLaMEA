import numpy as np

class QuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 100
        self.mutation_rate = 0.5
        self.crossover_rate = 0.9
        self.quantum_bits = 10
        self.probability = 0.05

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        for _ in range(self.budget):
            fitness = np.array([func(point) for point in population])
            best_points = population[np.argsort(fitness)]

            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = np.random.choice(best_points, size=2, replace=False)
                child = np.zeros(self.dim)
                for i in range(self.dim):
                    if np.random.rand() < self.crossover_rate:
                        child[i] = (parent1[i] + parent2[i]) / 2
                    else:
                        child = np.random.choice([parent1[i], parent2[i]], p=[self.probability, 1-self.probability])
                child = np.clip(child, self.lower_bound, self.upper_bound)
                if np.random.rand() < self.mutation_rate:
                    child = child + np.random.normal(0, 1, self.dim)
                new_population.append(child)

            population = np.array(new_population)

        return population[np.argmin(fitness)]

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
optimizer = QuantumDifferentialEvolution(budget, dim)
best_point = optimizer(func)
print("Best point:", best_point)
print("Minimum function value:", func(best_point))