import numpy as np

class HybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.temperature = 1.0
        self.cooling_rate = 0.995

    def _initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

    def _mutate(self, population, target_idx):
        indices = [idx for idx in range(self.pop_size) if idx != target_idx]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.mutation_factor * (b - c), -5.0, 5.0)
        return mutant

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _acceptance_probability(self, candidate_fitness, target_fitness):
        if candidate_fitness < target_fitness:
            return 1.0
        else:
            return np.exp((target_fitness - candidate_fitness) / self.temperature)

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])
        evals = self.pop_size

        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                target = population[i]
                mutant = self._mutate(population, i)
                trial = self._crossover(target, mutant)
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i] or np.random.rand() < self._acceptance_probability(trial_fitness, fitness[i]):
                    population[i] = trial
                    fitness[i] = trial_fitness

            self.temperature *= self.cooling_rate

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]