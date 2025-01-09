import numpy as np
import scipy.stats

class EnhancedAdaptiveDynamicScalingWithLevyFlight:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(12 * dim, 120)  # Adjusted population size for better exploration
        self.CR = 0.9
        self.F = 0.8
        self.current_evaluations = 0
        self.elitism_rate = 0.1  # New elitism rate for retaining top individuals

    def chaotic_map(self, x):
        return 4.0 * x * (1.0 - x)

    def levy_flight(self, scale=0.015):  # Adjusted scale for balanced exploration
        return scipy.stats.levy_stable.rvs(alpha=1.5, beta=0, scale=scale, size=self.dim)

    def initialize_population(self, bounds):
        lower, upper = bounds.lb, bounds.ub
        return np.random.rand(self.population_size, self.dim) * (upper - lower) + lower

    def evaluate(self, func, population):
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += len(population)
        return fitness

    def mutate(self, target_idx, bounds, population, diversity_factor):
        a, b, c = np.random.choice(np.delete(np.arange(self.population_size), target_idx), 3, replace=False)
        chaotic_factor = 1 + 0.1 * self.chaotic_map(np.random.rand())
        adaptive_F = self.F * diversity_factor * chaotic_factor
        adaptive_F *= np.random.uniform(0.9, 1.3)
        mutant_vector = population[a] + adaptive_F * (population[b] - population[c])
        levy_step = self.levy_flight(scale=0.015)  # Consistent scale
        mutant_vector += levy_step
        mutant_vector = np.clip(mutant_vector, bounds.lb, bounds.ub)
        return mutant_vector

    def crossover(self, target, mutant, diversity_factor):
        adaptive_CR = self.CR * (1.0 - diversity_factor)
        adaptive_CR *= np.random.uniform(0.8, 1.2)
        crossover_points = np.random.rand(self.dim) < adaptive_CR
        trial_vector = np.where(crossover_points, mutant, target)
        return trial_vector

    def optimize(self, func, bounds):
        chaotic_sequence = np.random.rand(self.population_size)
        population = self.initialize_population(bounds)
        fitness = self.evaluate(func, population)

        while self.current_evaluations < self.budget:
            new_population = []
            new_fitness = []

            diversity = np.std(population, axis=0).mean()
            diversity_factor = 1.0 / (1.0 + np.sqrt(diversity))

            elite_count = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_count]
            elites = population[elite_indices]

            for i in range(self.population_size - elite_count):  # Adjust loop for elitism
                target = population[i]
                mutant = self.mutate(i, bounds, population, diversity_factor)
                trial = self.crossover(target, mutant, diversity_factor)

                chaotic_sequence[i] = self.chaotic_map(chaotic_sequence[i])
                trial += chaotic_sequence[i] * (bounds.ub - bounds.lb) * np.random.uniform(0.01, 0.03)
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

            new_population.extend(elites)  # Include elites in the new population
            population = np.array(new_population)
            fitness = np.array(new_fitness + list(fitness[elite_indices]))

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def __call__(self, func):
        bounds = func.bounds
        best_solution, best_value = self.optimize(func, bounds)
        return best_solution, best_value