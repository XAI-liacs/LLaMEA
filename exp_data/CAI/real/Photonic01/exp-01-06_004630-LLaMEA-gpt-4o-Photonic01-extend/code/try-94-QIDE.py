import numpy as np
from scipy.optimize import minimize

class QIDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.cross_prob = 0.9
        self.diff_weight = 0.8
        self.success_history = []

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def quantum_superposition(self, population, lb, ub, func):
        # Adaptive beta based on success history
        beta = (0.18 / np.sqrt(self.dim)) * (1 + np.mean(self.success_history[-10:]))
        best_solution = population[np.argmin([func(ind) for ind in population])]
        quantum_population = population + beta * (best_solution - population) * np.random.normal(0, 1, (self.population_size, self.dim))
        np.clip(quantum_population, lb, ub, out=quantum_population)
        return quantum_population

    def local_search(self, individual, lb, ub, func):
        result = minimize(func, individual, bounds=[(lb[i], ub[i]) for i in range(self.dim)], method='L-BFGS-B')
        return result.x

    def differential_evolution(self, population, lb, ub, func):
        new_population = np.copy(population)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant_strategy = np.random.choice(['normal', 'best'])
            if mutant_strategy == 'best':
                best = population[np.argmin([func(ind) for ind in population])]
                mutant = np.clip(best + self.diff_weight * (b - c), lb, ub)
            else:
                mutant = np.clip(a + self.diff_weight * (b - c), lb, ub)
            diversity = np.std(population, axis=0).mean()
            recent_success_rate = np.mean(self.success_history[-10:]) if self.success_history else 0.5
            self.cross_prob = 0.5 + 0.4 * diversity * recent_success_rate
            cross_points = np.random.rand(self.dim) < (self.cross_prob * (0.5 + 0.5 * np.random.rand()))
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial = self.local_search(trial, lb, ub, func)
            if func(trial) < func(population[i]):
                new_population[i] = trial
                self.success_history.append(1)
            else:
                self.success_history.append(0)
        # Dynamic adjustment of population size
        self.population_size = min(self.initial_population_size, self.population_size + int(np.sum(self.success_history[-self.population_size:]) / self.population_size * self.dim))
        return new_population

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        evaluations = 0
        
        while evaluations < self.budget:
            quantum_population = self.quantum_superposition(population, lb, ub, func)
            population = self.differential_evolution(quantum_population, lb, ub, func)
            evaluations += self.population_size

        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]