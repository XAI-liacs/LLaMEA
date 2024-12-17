import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, int(0.1 * budget))
        self.mutation_factor = 0.85
        self.crossover_rate = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def levy_flight(self, lam=1.5):
        sigma = (np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2) / (np.math.gamma((1 + lam) / 2) * lam * 2 ** ((lam - 1) / 2))) ** (1 / lam)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / abs(v) ** (1 / lam)
        return step

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        while eval_count < self.budget:
            if eval_count > 0.4 * self.budget:
                new_size = max(5, int(self.population_size * 0.8))
                population = population[:new_size]
                fitness = fitness[:new_size]
                self.population_size = new_size

            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                mutation_factor = self.mutation_factor * (1 - np.std(fitness) / np.mean(fitness))

                best_individual = population[np.argmin(fitness)]
                mutant_vector = a + mutation_factor * (b - c) + 0.15 * (best_individual - a)
                if np.random.rand() < 0.5:  # Incorporate Lévy flights occasionally
                    mutant_vector += self.levy_flight()
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                self.crossover_rate = 0.7 + 0.3 * (1 - np.log(1 + eval_count / self.budget))

                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, population[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)

                trial_fitness = func(trial_vector)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

        return population[np.argmin(fitness)], np.min(fitness)