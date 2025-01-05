import numpy as np

class TDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.scaling_factor = 0.8
        self.crossover_rate = 0.9
        self.best_solution = None
        self.best_value = float('inf')

    def initialize_population(self, lb, ub):
        return lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def mutate(self, population, best_idx):
        mutants = np.zeros_like(population)
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)

            if np.random.rand() < 0.5:
                best_vector = population[best_idx]
            else:
                best_vector = population[a]

            mutants[i] = population[a] + self.scaling_factor * (population[b] - population[c])
            mutants[i] += 0.5 * (self.best_solution - best_vector)
        return mutants

    def crossover(self, target, mutant):
        trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, target)
        return trial

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            values = np.array([func(ind) for ind in population])
            evaluations += self.population_size

            best_idx = np.argmin(values)
            if values[best_idx] < self.best_value:
                self.best_value = values[best_idx]
                self.best_solution = population[best_idx].copy()

            mutants = self.mutate(population, best_idx)
            mutants = np.clip(mutants, lb, ub)

            new_population = []
            for i in range(self.population_size):
                trial = self.crossover(population[i], mutants[i])
                trial_value = func(trial)
                evaluations += 1
                if trial_value < values[i]:
                    new_population.append(trial)
                    if trial_value < self.best_value:
                        self.best_value = trial_value
                        self.best_solution = trial.copy()
                else:
                    new_population.append(population[i])

                if evaluations >= self.budget:
                    break

            population = np.array(new_population)
        
        return self.best_solution, self.best_value