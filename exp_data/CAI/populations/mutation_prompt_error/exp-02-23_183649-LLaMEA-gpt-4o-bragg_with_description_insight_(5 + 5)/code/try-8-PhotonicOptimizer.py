import numpy as np
from scipy.optimize import minimize

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = None
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.func_evals = 0

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.best_solution = self.population[0]

    def quasi_oppositional_initialization(self, lb, ub):
        new_population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        quasi_opposite_population = lb + ub - new_population
        combined_population = np.vstack((new_population, quasi_opposite_population))
        fitness_values = [self.evaluate(ind) for ind in combined_population]
        indices = np.argsort(fitness_values)[-self.population_size:]
        self.population = combined_population[indices]

    def mutation(self, target_idx):
        indices = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant_vector = self.population[a] + self.F * (self.population[b] - self.population[c]) + np.random.normal(0, 0.1, size=self.dim)  # Perturbation added
        return np.clip(mutant_vector, 0, 1)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        trial_vector = np.where(crossover_mask, mutant, target)
        return trial_vector

    def enforce_periodicity(self, vector, period):
        for i in range(0, self.dim, period):
            vector[i:i+period] = np.mean(vector[i:i+period])
        return vector

    def evaluate(self, individual):
        if self.func_evals >= self.budget:
            return float('-inf')
        self.func_evals += 1
        return self.func(individual)

    def optimize(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub
        self.quasi_oppositional_initialization(lb, ub)

        while self.func_evals < self.budget:
            for i in range(self.population_size):
                target = self.population[i]
                mutant = self.mutation(i)
                trial = self.crossover(target, mutant)
                trial = self.enforce_periodicity(trial, period=2)
                trial_fitness = self.evaluate(trial)
                if trial_fitness > self.evaluate(target):
                    self.population[i] = trial
                    if trial_fitness > self.best_fitness:
                        self.best_solution = trial
                        self.best_fitness = trial_fitness

        # Local optimization using BFGS for fine-tuning
        result = minimize(lambda x: -self.evaluate(x), self.best_solution, bounds=[(lb[i], ub[i]) for i in range(self.dim)], method='L-BFGS-B')
        if result.success:
            self.best_solution = result.x
            self.best_fitness = -result.fun

    def __call__(self, func):
        self.optimize(func)
        return self.best_solution