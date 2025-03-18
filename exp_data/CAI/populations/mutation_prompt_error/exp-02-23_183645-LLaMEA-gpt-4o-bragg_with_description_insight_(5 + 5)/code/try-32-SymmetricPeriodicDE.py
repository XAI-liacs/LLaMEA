import numpy as np
from scipy.optimize import minimize

class SymmetricPeriodicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.cross_prob = 0.7
        self.diff_weight = 0.8
        self.local_optimization_prob = 0.1
        self.population_growth_factor = 1.1
        self.diversity_threshold = 0.1

    def _initialize_population(self, lb, ub, population_size):
        mid_point = (lb + ub) / 2
        half_range = (ub - lb) / 2
        return mid_point + np.random.uniform(-half_range, half_range, (population_size, self.dim))

    def _apply_periodicity_promotion(self, candidate):
        period = np.random.choice([1, 3, 5])  # Adjusted periods to better target periodic structures
        for i in range(self.dim):
            candidate[i] = candidate[i % period]
        return candidate
    
    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])
    
    def _calculate_population_diversity(self, population):
        return np.mean(np.std(population, axis=0))

    def _adaptive_crossover(self, diversity):
        if diversity < self.diversity_threshold:
            return min(1.0, self.cross_prob + 0.2)  # Increase crossover probability if diversity is low
        return self.cross_prob

    def _differential_evolution_step(self, population, scores, lb, ub, func):
        diversity = self._calculate_population_diversity(population)
        crossover_probability = self._adaptive_crossover(diversity)
        
        for i in range(len(population)):
            indices = [idx for idx in range(len(population)) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            
            mutant = np.clip(population[a] + self.diff_weight * (population[b] - population[c]), lb, ub)
            
            cross_points = np.random.rand(self.dim) < crossover_probability
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