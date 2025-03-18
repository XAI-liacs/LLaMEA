import numpy as np
from scipy.optimize import minimize

class DynamicSymbioticEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.scale_factor = 0.8
        self.crossover_probability = 0.7
        self.periodicity_weight = 0.2
        self.pso_weight = 0.5
        self.velocity_clamp = 0.1
        
    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def initialize_velocity(self):
        return np.random.uniform(-self.velocity_clamp, self.velocity_clamp, (self.population_size, self.dim))
    
    def periodicity_cost(self, solution):
        periodicity_error = np.sum((solution - np.roll(solution, 2))**2)
        return periodicity_error

    def differential_evolution_step(self, population, lb, ub, func):
        new_population = np.empty_like(population)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.scale_factor * (b - c), lb, ub)
            cross_points = np.random.rand(self.dim) < self.crossover_probability
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial_with_periodicity = trial + self.periodicity_weight * self.periodicity_cost(trial)

            if func(trial_with_periodicity) < func(population[i]):
                new_population[i] = trial_with_periodicity
            else:
                new_population[i] = population[i]
        
        return new_population
    
    def pso_step(self, population, velocity, personal_best, global_best, lb, ub):
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        self.pso_weight = np.random.uniform(0.4, 0.9)  # Dynamically adjust PSO weight
        velocity = self.velocity_clamp * (velocity 
                                          + self.pso_weight * r1 * (personal_best - population) 
                                          + self.pso_weight * r2 * (global_best - population))
        population = np.clip(population + velocity, lb, ub)
        return population, velocity
    
    def local_optimization(self, x0, func, lb, ub):
        res = minimize(func, x0, method='L-BFGS-B', bounds=[(lb[i], ub[i]) for i in range(self.dim)])
        return res.x if res.success else x0
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        velocity = self.initialize_velocity()
        personal_best = population.copy()
        personal_best_scores = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_scores)]
        best_solution = None
        best_score = float('inf')
        
        for _ in range(self.budget // self.population_size):
            population = self.differential_evolution_step(population, lb, ub, func)
            population, velocity = self.pso_step(population, velocity, personal_best, global_best, lb, ub)
            for i in range(self.population_size):
                population[i] = self.local_optimization(population[i], func, lb, ub)
                score = func(population[i])
                
                if score < personal_best_scores[i]:
                    personal_best[i] = population[i]
                    personal_best_scores[i] = score
                
                if score < best_score:
                    best_score = score
                    best_solution = population[i]
                    global_best = best_solution
        
        return best_solution