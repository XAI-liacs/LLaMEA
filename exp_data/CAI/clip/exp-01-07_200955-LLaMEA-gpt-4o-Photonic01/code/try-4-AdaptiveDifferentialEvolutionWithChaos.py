import numpy as np

class AdaptiveDifferentialEvolutionWithChaos:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(10 * dim, 100)  # Adaptive population size
        self.CR = 0.9  # Crossover probability
        self.F = 0.8   # Differential weight
        self.current_evaluations = 0

    def chaotic_map(self, x):
        # Logistic map for chaotic sequence generation
        return 4.0 * x * (1.0 - x)

    def initialize_population(self, bounds):
        lower, upper = bounds.lb, bounds.ub
        return np.random.rand(self.population_size, self.dim) * (upper - lower) + lower

    def evaluate(self, func, population):
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += len(population)
        return fitness

    def select_parents(self, population, fitness):
        idx = np.random.choice(len(population), 3, replace=False)
        while len(set(idx)) < 3:  # Ensure unique parents
            idx = np.random.choice(len(population), 3, replace=False)
        return population[idx], fitness[idx]

    def mutate(self, target_idx, bounds, population):
        a, b, c = np.random.choice(np.delete(np.arange(self.population_size), target_idx), 3, replace=False)
        target = population[target_idx]
        mutant_vector = population[a] + self.F * (population[b] - population[c])
        mutant_vector = np.clip(mutant_vector, bounds.lb, bounds.ub)
        return mutant_vector

    def crossover(self, target, mutant):
        crossover_points = np.random.rand(self.dim) < self.CR
        trial_vector = np.where(crossover_points, mutant, target)
        return trial_vector

    def optimize(self, func, bounds):
        # Initialize chaotic sequences
        chaotic_sequence = np.random.rand(self.population_size)

        # Initialize population
        population = self.initialize_population(bounds)
        fitness = self.evaluate(func, population)

        while self.current_evaluations < self.budget:
            new_population = []
            new_fitness = []

            for i in range(self.population_size):
                target = population[i]
                mutant = self.mutate(i, bounds, population)
                trial = self.crossover(target, mutant)

                # Enhance trial vector with chaotic sequence
                chaotic_sequence[i] = self.chaotic_map(chaotic_sequence[i])
                trial = trial + chaotic_sequence[i] * (bounds.ub - bounds.lb) * 0.05
                trial = np.clip(trial, bounds.lb, bounds.ub)

                trial_fitness = func(trial)
                self.current_evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    new_fitness.append(trial_fitness)
                else:
                    new_population.append(target)
                    new_fitness.append(fitness[i])

                if self.current_evaluations >= self.budget:
                    break

            population = np.array(new_population)
            fitness = np.array(new_fitness)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def __call__(self, func):
        bounds = func.bounds
        best_solution, best_value = self.optimize(func, bounds)
        return best_solution, best_value