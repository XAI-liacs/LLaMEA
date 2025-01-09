import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 5 * dim
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.evaluations = 0

    def __call__(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        self.evaluations += self.pop_size

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break

                candidates = list(range(0, i)) + list(range(i + 1, self.pop_size))
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                crossover = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True

                trial = np.where(crossover, mutant, self.population[i])
                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

            # Adaptive population size adjustment based on diversity
            diversity = np.std(self.population, axis=0).mean()
            if diversity < 1e-5:
                self.pop_size = max(4, self.pop_size // 2)
            elif diversity > 0.1:
                self.pop_size = min(5 * self.dim, self.pop_size * 2)
            
            # Re-evaluate the population size
            if self.pop_size != len(self.population):
                self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
                self.fitness = np.array([func(ind) for ind in self.population])
                self.evaluations += self.pop_size - len(self.population)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]

# Example usage:
# optimizer = AdaptiveDifferentialEvolution(budget=10000, dim=10)
# best_solution = optimizer(some_black_box_function)
# print(best_solution)