import numpy as np
from scipy.optimize import minimize

class HybridEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.best_solution = None
        self.best_score = float('inf')
    
    def quasi_oppositional_initialization(self, lb, ub):
        pop = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        q_opposite_pop = ub + lb - pop
        combined_pop = np.vstack((pop, q_opposite_pop))
        periodic_bias = (np.arange(self.dim) % 2) * (ub - lb) / 4 + lb  # Added bias towards periodic solution
        combined_pop += periodic_bias
        return combined_pop[np.random.choice(combined_pop.shape[0], self.population_size, replace=False)]

    def differential_evolution(self, func, lb, ub):
        population = self.quasi_oppositional_initialization(lb, ub)
        scores = np.array([func(ind) for ind in population])
        evaluations = len(population)
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = [index for index in range(self.population_size) if index != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < self.crossover_probability
                trial = np.where(crossover, mutant, population[i])
                
                trial_score = func(trial)
                evaluations += 1
                
                if trial_score < scores[i]:
                    population[i] = trial
                    scores[i] = trial_score
                    
                    if trial_score < self.best_score:
                        self.best_solution = trial
                        self.best_score = trial_score
                
                if evaluations >= self.budget:
                    break
        return self.best_solution

    def local_optimization(self, func, initial_solution, lb, ub):
        result = minimize(func, initial_solution, bounds=list(zip(lb, ub)), method='L-BFGS-B')
        return result.x, result.fun

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        solution = self.differential_evolution(func, lb, ub)
        if solution is not None:
            solution, score = self.local_optimization(func, solution, lb, ub)
            if score < self.best_score:
                self.best_solution = solution
                self.best_score = score
        return self.best_solution