import numpy as np
from scipy.optimize import minimize

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.95  # Crossover probability (adjusted for better exploration)
        self.lb = None
        self.ub = None

    def quasi_oppositional_initialization(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        opposite_pop = lb + ub - pop
        combined_pop = np.vstack((pop, opposite_pop))
        fitness = np.array([self.evaluate(ind) for ind in combined_pop])
        indices = fitness.argsort()[:self.population_size]
        return combined_pop[indices], fitness[indices]
    
    def evaluate(self, x):
        if hasattr(self, 'evaluations') and self.evaluations >= self.budget:
            return np.inf
        self.evaluations += 1
        return self.func(x)

    def de_step(self, population, fitness):
        new_population = np.zeros_like(population)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            adapted_F = 0.4 + 0.1 * np.random.rand()  # Adaptive differential weight
            mutant = np.clip(a + adapted_F * (b - c), self.lb, self.ub)
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial_fitness = self.evaluate(trial)
            if trial_fitness < fitness[i]:
                new_population[i] = trial
                fitness[i] = trial_fitness
            else:
                new_population[i] = population[i]
        return new_population, fitness

    def apply_periodicity_constraints(self, x):
        period = int(self.dim / 2)  # Assume a 2-layer period 
        x[:period] = x[:period] * (self.dim // period)  # Repeat pattern
        return x

    def local_optimization(self, x):
        res = minimize(self.func, x, method='L-BFGS-B', bounds=list(zip(self.lb, self.ub)))
        return res.x if res.success else x

    def __call__(self, func):
        self.func = func
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.evaluations = 0
        population, fitness = self.quasi_oppositional_initialization(func.bounds)
        
        while self.evaluations < self.budget:
            population, fitness = self.de_step(population, fitness)
            for i in range(self.population_size):
                if np.random.rand() < 0.1:  # Occasionally apply local optimization
                    x_periodic = self.apply_periodicity_constraints(population[i].copy())
                    population[i] = self.local_optimization(x_periodic)
                    fitness[i] = self.evaluate(population[i])

        best_idx = np.argmin(fitness)
        return population[best_idx]