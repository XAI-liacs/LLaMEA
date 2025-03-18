import numpy as np
from scipy.optimize import minimize

class SymmetricPeriodicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20  # Initial population size
        self.cross_prob = 0.7  # Crossover probability
        self.diff_weight = 0.8  # Differential weight
        self.local_optimization_prob = 0.1  # Probability to apply local optimization
        self.population_growth_factor = 1.1  # Factor to grow population size

    def _initialize_population(self, lb, ub, population_size):
        mid_point = (lb + ub) / 2
        half_range = (ub - lb) / 2
        return mid_point + np.random.uniform(-half_range, half_range, (population_size, self.dim))
    
    def _apply_periodicity_promotion(self, candidate):
        period = np.random.choice([1, 2, 4])  # Adjusted periods to target specific structures
        for i in range(self.dim):
            candidate[i] = candidate[i % period]
        return candidate
    
    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])
    
    def _differential_evolution_step(self, population, scores, lb, ub, func):
        avg_score = np.mean(scores)  # Calculate average score
        for i in range(len(population)):
            indices = [idx for idx in range(len(population)) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            
            mutant = np.clip(population[a] + self.diff_weight * (population[b] - population[c]), lb, ub)
            
            # Adjust crossover probability based on evaluation progress
            progress_ratio = (self.budget - len(population) * len(scores)) / self.budget
            self.cross_prob = 0.5 + 0.5 * progress_ratio

            population_diversity = np.std(population)
            self.diff_weight = 0.5 + 0.5 * (population_diversity / np.mean([ub - lb]))

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
        population_size = self.initial_population_size
        population = self._initialize_population(lb, ub, population_size)
        scores = self._evaluate_population(population, func)
        
        evaluations = population_size
        while evaluations < self.budget:
            # Dynamic population resizing
            if evaluations > self.budget // 2:
                population_size = int(self.population_growth_factor * population_size)
                population = np.resize(population, (population_size, self.dim))
                scores = np.resize(scores, population_size)
                for i in range(len(scores)):
                    if scores[i] == 0:
                        scores[i] = func(population[i])
                        evaluations += 1

            population, scores = self._differential_evolution_step(population, scores, lb, ub, func)
            evaluations += len(population)
            
            if np.random.rand() < self.local_optimization_prob:
                idx = np.random.randint(0, len(population))
                candidate = population[idx]
                optimized = self._local_optimization(candidate, func, lb, ub)
                optimized_score = func(optimized)
                evaluations += 1
                if optimized_score > scores[idx]:
                    population[idx] = optimized
                    scores[idx] = optimized_score
        
        best_idx = np.argmax(scores)
        return population[best_idx]