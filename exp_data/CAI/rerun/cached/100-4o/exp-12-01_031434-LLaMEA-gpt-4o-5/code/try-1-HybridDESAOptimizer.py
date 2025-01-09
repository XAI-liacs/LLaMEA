import numpy as np

class HybridDESAOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + int(5 * np.sqrt(dim))
        self.temperature = 100.0
        self.cooling_rate = 0.95

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def _mutate(self, target_idx, population):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        F = 0.8  # Differential weight
        mutant = a + F * (b - c)
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def _crossover(self, target, mutant):
        crossover_rate = 0.9
        crossover_mask = np.random.rand(self.dim) < crossover_rate
        offspring = np.where(crossover_mask, mutant, target)
        return offspring

    def _acceptance_probability(self, current_cost, new_cost):
        if new_cost < current_cost:
            return 1.0
        else:
            return np.exp((current_cost - new_cost) / self.temperature)

    def __call__(self, func):
        population = self._initialize_population()
        fitness = self._evaluate_population(population, func)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        evaluations = self.population_size

        while evaluations < self.budget:
            for target_idx in range(self.population_size):
                mutant = self._mutate(target_idx, population)
                offspring = self._crossover(population[target_idx], mutant)
                off_fitness = func(offspring)
                evaluations += 1

                if off_fitness < fitness[target_idx]:
                    population[target_idx] = offspring
                    fitness[target_idx] = off_fitness
                else:
                    ap = self._acceptance_probability(fitness[target_idx], off_fitness)
                    if np.random.rand() < ap:
                        population[target_idx] = offspring
                        fitness[target_idx] = off_fitness

                if off_fitness < best_fitness:
                    best_solution = offspring
                    best_fitness = off_fitness

                if evaluations >= self.budget:
                    break

            self.temperature *= self.cooling_rate

        return best_solution