import numpy as np
from scipy.optimize import minimize

class BraggMirrorOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20  # Set a reasonable population size
        self.cross_prob = 0.7  # Crossover probability
        self.diff_weight = 0.8  # Differential weight
        self.local_optimization_prob = 0.1  # Probability to apply local optimization
    
    def _initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def _apply_periodicity_promotion(self, candidate, diversity):
        period = 2 + int(diversity > 0.1)  # Adaptive period based on diversity
        for i in range(self.dim):
            candidate[i] = candidate[i % period]
        return candidate
    
    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])
    
    def _differential_evolution_step(self, population, scores, lb, ub, func):
        for i in range(self.population_size):
            # Select three random indices different from i
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            
            # Mutation
            mutant = np.clip(population[a] + self.diff_weight * (population[b] - population[c]), lb, ub)
            
            # Crossover
            cross_points = np.random.rand(self.dim) < self.cross_prob
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            # Periodicity promotion
            diversity = np.std(population, axis=0).mean()
            trial = self._apply_periodicity_promotion(trial, diversity)
            
            # Selection
            trial_score = func(trial)
            if trial_score > scores[i]:
                population[i] = trial
                scores[i] = trial_score
        
        return population, scores
    
    def _local_optimization(self, candidate, func, lb, ub):
        def local_func(x):
            return -func(x)  # Minimize negative reflectivity for maximization

        result = minimize(local_func, candidate, bounds=[(lb[j], ub[j]) for j in range(self.dim)],
                          method='L-BFGS-B', options={'maxiter': 10})
        return result.x if result.success else candidate
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self._initialize_population(lb, ub)
        scores = self._evaluate_population(population, func)
        
        evaluations = self.population_size
        while evaluations < self.budget:
            population, scores = self._differential_evolution_step(population, scores, lb, ub, func)
            evaluations += self.population_size
            
            # Local optimization step
            if np.random.rand() < self.local_optimization_prob:
                idx = np.random.randint(0, self.population_size)
                candidate = population[idx]
                optimized = self._local_optimization(candidate, func, lb, ub)
                optimized_score = func(optimized)
                evaluations += 1
                if optimized_score > scores[idx]:
                    population[idx] = optimized
                    scores[idx] = optimized_score
        
        best_idx = np.argmax(scores)
        return population[best_idx]