import numpy as np
from scipy.optimize import minimize

class QuantumInspiredExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 * dim
        self.mutation_factor = 0.85
        self.crossover_rate = 0.85
        self.evaluations = 0

    def quantum_initialization(self, bounds):
        lb, ub = bounds
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        return population

    def quantum_state_update(self, individual, global_best):
        quantum_prob = np.random.rand(self.dim)
        phase = np.random.rand(self.dim) * 2 * np.pi
        updated_state = individual + self.mutation_factor * np.sin(phase) * (global_best - individual) * quantum_prob
        return updated_state

    def periodicity_enhancement(self, trial):
        if np.random.rand() < 0.5:
            period = np.random.randint(1, self.dim // 2 + 1)
            trial = np.tile(trial[:period], self.dim // period + 1)[:self.dim]
        return trial

    def hybrid_quantum_evolution(self, func, bounds):
        population = self.quantum_initialization(bounds)
        global_best_solution = None
        global_best_score = float('-inf')

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[idxs]

                mutant = a + self.mutation_factor * (b - c)
                mutant = np.clip(mutant, *bounds)

                trial = self.quantum_state_update(mutant, global_best_solution if global_best_solution is not None else a)
                trial = self.periodicity_enhancement(trial)
                trial = np.clip(trial, *bounds)

                score = func(trial)
                self.evaluations += 1

                if score > func(population[i]):
                    population[i] = trial
                    if score > global_best_score:
                        global_best_solution, global_best_score = trial, score

        return global_best_solution

    def gradient_based_refinement(self, func, best_solution, bounds):
        tol_factor = max(1, self.budget - self.evaluations)
        res = minimize(lambda x: -func(x), best_solution, bounds=bounds, method='L-BFGS-B', options={'ftol': 1e-6 / tol_factor})
        return res.x

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        best_solution = self.hybrid_quantum_evolution(func, bounds)
        best_solution = self.gradient_based_refinement(func, best_solution, bounds)
        return best_solution