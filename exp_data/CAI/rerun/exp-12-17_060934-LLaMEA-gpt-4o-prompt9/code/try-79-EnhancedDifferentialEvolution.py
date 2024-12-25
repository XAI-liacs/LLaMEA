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

                # Adaptive mutation factor with fitness history
                prev_best = np.min(fitness)
                mut_factor_change = 0.1 * (prev_best - np.min(fitness)) / (np.abs(prev_best) + 1e-30)
                mutation_factor = self.mutation_factor * (1 + mut_factor_change)

                scaling_factor = np.random.uniform(0.9, 1.1)
                best_individual = population[np.argmin(fitness)]
                elite_guided_factor = 0.15 * (best_individual - population.mean(axis=0))
                
                dynamic_bias = 0.1 * np.sign(np.mean(fitness) - fitness[i]) * np.abs(best_individual - a)
                mutant_vector = a + mutation_factor * scaling_factor * (b - c) + elite_guided_factor + dynamic_bias
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                # Adaptive crossover rate based on fitness improvements
                improvement = (np.min(fitness) - np.mean(fitness)) / (np.std(fitness) + 1e-30)
                self.crossover_rate = 0.5 + 0.4 * np.tanh(improvement)
                
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, population[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)

                trial_fitness = func(trial_vector)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

        return population[np.argmin(fitness)], np.min(fitness)