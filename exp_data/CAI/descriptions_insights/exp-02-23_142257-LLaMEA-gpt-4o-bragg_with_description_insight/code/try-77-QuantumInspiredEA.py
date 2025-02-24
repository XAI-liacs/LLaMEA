import numpy as np
from scipy.optimize import minimize

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.population = None
        self.best_solution = None
        self.best_score = float('-inf')
        self.eval_count = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.best_solution = self.population[0]

    def quantum_superposition(self):
        for i in range(self.population_size):
            if self.eval_count >= self.budget:
                break
            phi = np.random.uniform(0, 2 * np.pi, self.dim)
            superposed_state = np.where(np.random.rand(self.dim) < 0.5,
                                        self.population[i] * np.cos(phi),
                                        self.population[i] * np.sin(phi))
            self.population[i] = np.clip(superposed_state, lb, ub)

    def quantum_entanglement(self, func):
        entangled_pop = np.zeros_like(self.population)
        for i in range(self.population_size):
            if self.eval_count >= self.budget:
                break
            partner_idx = np.random.randint(self.population_size)
            partner = self.population[partner_idx]
            entangled_pop[i] = 0.5 * (self.population[i] + partner)  # Simple entanglement
            score = func(entangled_pop[i])
            self.eval_count += 1
            
            if score > self.best_score:
                self.best_score = score
                self.best_solution = entangled_pop[i]

        self.population = entangled_pop

    def local_search(self, func, lb, ub):
        result = minimize(func, self.best_solution, bounds=np.c_[lb, ub], method='L-BFGS-B')
        self.eval_count += result.nfev

        if result.fun > self.best_score:
            self.best_score = result.fun
            self.best_solution = result.x

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)

        while self.eval_count < self.budget:
            self.quantum_superposition()
            self.quantum_entanglement(func)
            self.local_search(func, lb, ub)

        return self.best_solution