import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, int(0.1 * budget))
        self.mutation_factor = 0.85
        self.crossover_rate = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        while eval_count < self.budget:
            if eval_count > 0.3 * self.budget:
                new_size = max(5, int(self.population_size * 0.75))
                population = population[:new_size]
                fitness = fitness[:new_size]
                self.population_size = new_size

            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                mutation_factor1 = self.mutation_factor * ((self.budget - eval_count) / self.budget) * (1 - np.std(fitness) / (np.mean(fitness) + 1e-30))
                mutation_factor2 = self.mutation_factor * (1 - (np.var(fitness) / (np.mean(fitness**2) + 1e-30)))
                mutation_factor = (mutation_factor1 + mutation_factor2) / 2 * (1 + 0.1 * (np.std(population, axis=0) / (np.mean(population, axis=0) + 1e-30)).mean())

                # Introduced probabilistic scaling mechanism
                scaling_factor = np.random.uniform(0.9, 1.1) * np.exp(-0.05 * np.abs(np.min(fitness) - np.mean(fitness)))
                mutation_factor *= scaling_factor * 1.022

                # Slight adjustment to enhance exploration and selection
                mutation_factor *= (0.98 + 0.02 * np.random.rand())

                best_individual = population[np.argmin(fitness)]
                elite_guided_factor = 0.1 * (best_individual - population.mean(axis=0))
                mutant_vector = a + mutation_factor * (b - c) + 0.15 * (best_individual - a) + elite_guided_factor
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                diversity_factor = (np.std(population, axis=0).mean() / (np.std(population) + 1e-30)) * (np.max(fitness) - np.min(fitness)) / (np.mean(fitness) + 1e-30)
                self.crossover_rate = 0.65 + 0.35 * (1 - (np.min(fitness) / (np.mean(fitness) + 1e-30))) * diversity_factor
                self.crossover_rate *= 1.025

                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, population[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)

                trial_fitness = func(trial_vector)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

        return population[np.argmin(fitness)], np.min(fitness)