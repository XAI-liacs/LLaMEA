import numpy as np
from scipy.optimize import minimize

class SymmetricPeriodicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20  # Set a reasonable population size
        self.cross_prob = 0.7  # Crossover probability
        self.diff_weight = 0.8  # Differential weight
        self.local_optimization_prob = 0.1  # Initial probability to apply local optimization
    
    def _initialize_population(self, lb, ub):
        mid_point = (lb + ub) / 2
        half_range = (ub - lb) / 2
        return mid_point + np.random.uniform(-half_range, half_range, (self.population_size, self.dim))
    
    def _apply_periodicity_promotion(self, candidate):
        period = np.random.randint(1, 4)  # Adaptive period selection
        for i in range(self.dim):
            candidate[i] = candidate[i % period]
        return candidate
    
    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])
    
    def _differential_evolution_step(self, population, scores, lb, ub, func):
        for i in range(self.population_size):
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            
            mutant = np.clip(population[a] + self.diff_weight * (population[b] - population[c]), lb, ub)
            
            cross_points = np.random.rand(self.dim) < self.cross_prob
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            trial = self._apply_periodicity_promotion(trial)
            
            trial_score = func(trial)
            if trial_score > scores[i]:
                population[i] = trial
                scores[i] = trial_score
        
        return population, scores
    
    def _local_optimization(self, candidate, func, lb, ub):
        def local_func(x):
            return -func(x)

        result = minimize(local_func, candidate, bounds=[(lb[j], ub[j]) for j in range(self.dim)],
                          method='L-BFGS-B', options={'maxiter': 10})
        return result.x if result.success else candidate
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self._initialize_population(lb, ub)
        scores = self._evaluate_population(population, func)
        
        evaluations = self.population_size
        while evaluations < self.budget:
            diversity = np.std(scores) / np.mean(scores)
            self.diff_weight = max(0.5, min(1.0, diversity))  # Adjust differential weight adaptively based on diversity
            self.population_size = max(10, int(20 * diversity))  # Dynamic population size based on diversity
            population, scores = self._differential_evolution_step(population, scores, lb, ub, func)
            evaluations += self.population_size
            
            self.local_optimization_prob = max(0.05, min(0.3, diversity))
            
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