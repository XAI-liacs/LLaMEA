import numpy as np
import scipy.optimize as opt

class AdaptiveQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 * dim  # Increased population size
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
        theta = np.random.uniform(-np.pi, np.pi, self.dim)  # Broadened angle range
        rotation_matrix = np.exp(1j * theta)
        shifted_individual = individual * rotation_matrix.real
        return np.clip(shifted_individual, lb, ub)

    def mutation(self, individual, lb, ub, iteration):
        # Adaptive mutation based on the iteration
        perturbation_scale = 0.1 * (1 - iteration / self.budget)
        perturbation = np.random.normal(0, perturbation_scale, self.dim)
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
        iteration = 0
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                individual = self.pop[i]
                new_individual = self.quantum_gate_operation(individual, lb, ub)
                trial = self.mutation(new_individual, lb, ub, iteration)
                
                trial_score = func(trial)
                self.evaluations += 1
                
                if trial_score < scores[i]:
                    scores[i] = trial_score
                    self.pop[i] = trial
                    if trial_score < self.best_score:
                        self.best_score = trial_score
                        self.best_solution = trial
            iteration += 1
            if self.evaluations < self.budget:
                self.local_search(func, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        scores = self.evaluate_population(func)
        self.optimize(func, scores, lb, ub)
        return self.best_solution