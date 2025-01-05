import numpy as np

class HGDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_rate = 0.7
        self.mutation_factor = 0.5

    def initialize_population(self, lb, ub):
        return lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def select_parents(self, population):
        return population[np.random.choice(self.population_size, 3, replace=False)]

    def mutation(self, target, donors, lb, ub):
        mutant = donors[0] + self.mutation_factor * (donors[1] - donors[2])
        return np.clip(mutant, lb, ub)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        return np.where(crossover_mask, mutant, target)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        population = self.initialize_population(lb, ub)
        best_solution = None
        best_value = float('inf')

        while evaluations < self.budget:
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                target = population[i]
                donors = self.select_parents(population)
                mutant = self.mutation(target, donors, lb, ub)
                trial = self.crossover(target, mutant)
                
                trial_value = func(trial)
                evaluations += 1
                
                if trial_value < func(target):
                    new_population[i] = trial
                else:
                    new_population[i] = target

                if trial_value < best_value:
                    best_value = trial_value
                    best_solution = trial.copy()

                if evaluations >= self.budget:
                    break

            population = new_population

        return best_solution, best_value