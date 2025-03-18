import numpy as np
import scipy.optimize as opt

class AdaptiveQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.evaluations = 0
        self.best_solution = None
        self.best_score = float('inf')
        self.pop = None
    
    def initialize_population(self, lb, ub):
        self.pop = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def evaluate_population(self, func):
        scores = np.apply_along_axis(func, 1, self.pop)
        self.evaluations += len(scores)
        return scores

    def quantum_gate_operation(self, individual, lb, ub):
        # Apply a simple quantum-inspired rotation using a random angle
        theta = np.random.uniform(-np.pi / 2, np.pi / 2, self.dim)
        rotation_matrix = np.cos(theta) + np.sin(theta) * 1j
        shifted_individual = individual * rotation_matrix.real
        return np.clip(shifted_individual, lb, ub)

    def mutation(self, individual, lb, ub):
        # Introduce small random changes
        perturbation = np.random.normal(0, 0.1, self.dim)
        mutated_individual = individual + perturbation
        return np.clip(mutated_individual, lb, ub)

    def local_search(self, func, lb, ub):
        if self.best_solution is not None:
            result = opt.minimize(func, self.best_solution, bounds=list(zip(lb, ub)), method='L-BFGS-B')
            if result.fun < self.best_score:
                self.best_score = result.fun
                self.best_solution = result.x
                self.evaluations += result.nfev

    def optimize(self, func, scores, lb, ub):
        for i in range(self.population_size):
            individual = self.pop[i]
            # Apply quantum gate operation
            new_individual = self.quantum_gate_operation(individual, lb, ub)
            # Apply mutation
            trial = self.mutation(new_individual, lb, ub)

            trial_score = func(trial)
            self.evaluations += 1
            if trial_score < scores[i]:
                scores[i] = trial_score
                self.pop[i] = trial
                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best_solution = trial

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        scores = self.evaluate_population(func)
        while self.evaluations < self.budget:
            self.optimize(func, scores, lb, ub)
            if self.evaluations < self.budget:
                self.local_search(func, lb, ub)
        return self.best_solution