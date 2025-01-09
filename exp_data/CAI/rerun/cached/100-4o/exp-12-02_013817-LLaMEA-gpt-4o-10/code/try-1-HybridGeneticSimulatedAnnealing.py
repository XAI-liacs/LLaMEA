import numpy as np

class HybridGeneticSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.mutation_rate = 0.1
        self.temperature = 1.0
        self.alpha = 0.99

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def _select_parents(self, population, fitness):
        fitness += abs(fitness.min()) + 1e-8  # Avoid negative probabilities
        probabilities = fitness / fitness.sum()
        return population[np.random.choice(self.population_size, 2, p=probabilities)]

    def _crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, self.dim)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def _mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            mutation_strength = np.random.normal(0, 0.2, self.dim)  # Gaussian mutation
            individual += mutation_strength
            individual = np.clip(individual, self.lower_bound, self.upper_bound)
        return individual

    def _anneal(self, candidate, best, func):
        candidate_fitness = func(candidate)
        best_fitness = func(best)
        if candidate_fitness < best_fitness:
            return candidate, candidate_fitness
        else:
            acceptance_probability = np.exp((best_fitness - candidate_fitness) / self.temperature)
            if np.random.rand() < acceptance_probability:
                return candidate, candidate_fitness
        return best, best_fitness

    def __call__(self, func):
        population = self._initialize_population()
        fitness = self._evaluate_population(population, func)
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]
        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = []
            for _ in range(self.population_size // 2):
                parents = self._select_parents(population, fitness)
                child1, child2 = self._crossover(*parents)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.extend([child1, child2])
                
            new_population = np.array(new_population)
            new_fitness = self._evaluate_population(new_population, func)
            evaluations += self.population_size

            for i in range(self.population_size):
                best_solution, best_fitness = self._anneal(new_population[i], best_solution, func)
            
            population = new_population
            fitness = new_fitness
            self.temperature *= self.alpha

        return best_solution