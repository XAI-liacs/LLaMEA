import numpy as np
from scipy.optimize import minimize

class MultiPhasePhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Heuristic for population size
        self.bounds = None
        
    def initialize_population(self):
        lb, ub = self.bounds.lb, self.bounds.ub
        population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        oppositional_population = lb + ub - population
        return np.vstack((population, oppositional_population))
    
    def adaptive_swarm_update(self, population, velocities, personal_best_positions, global_best_position):
        inertia = 0.5 + 0.1 * np.random.rand()
        cognitive_coeff = 1.5
        social_coeff = 1.5
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)
        
        velocities = (inertia * velocities +
                     cognitive_coeff * r1 * (personal_best_positions - population) +
                     social_coeff * r2 * (global_best_position - population))
        
        new_population = np.clip(population + velocities, self.bounds.lb, self.bounds.ub)
        return new_population, velocities
    
    def periodicity_penalty(self, solution):
        period = self.dim // 2
        penalties = [(solution[i] - solution[i + period]) ** 2 for i in range(self.dim - period)]
        return np.sum(penalties)
    
    def __call__(self, func):
        self.bounds = func.bounds
        population = self.initialize_population()
        velocities = np.zeros_like(population)
        personal_best_positions = population.copy()
        personal_best_scores = np.full(self.population_size, float('inf'))
        
        global_best_position = None
        global_best_score = float('inf')
        
        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                candidate = population[i]
                candidate_score = func(candidate) + self.periodicity_penalty(candidate)
                eval_count += 1
                
                if candidate_score < personal_best_scores[i]:
                    personal_best_scores[i] = candidate_score
                    personal_best_positions[i] = candidate
                
                if candidate_score < global_best_score:
                    global_best_score = candidate_score
                    global_best_position = candidate
            
            population, velocities = self.adaptive_swarm_update(population, velocities, personal_best_positions, global_best_position)
            
            # Local optimization using L-BFGS-B if budget allows
            if eval_count < self.budget:
                result = minimize(func, global_best_position, method='L-BFGS-B', bounds=zip(self.bounds.lb, self.bounds.ub))
                eval_count += result.nfev
                if result.fun < global_best_score:
                    global_best_score = result.fun
                    global_best_position = result.x
        
        return global_best_position