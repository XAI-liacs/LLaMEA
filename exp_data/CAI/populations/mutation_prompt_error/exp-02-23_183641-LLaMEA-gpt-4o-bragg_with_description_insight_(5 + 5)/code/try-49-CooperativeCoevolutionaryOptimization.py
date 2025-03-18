import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor

class CooperativeCoevolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.6
        self.crossover_rate = 0.9
        self.current_budget = 0
        self.surrogate_model = RandomForestRegressor(n_estimators=10)
        
    def _initialize_population(self, bounds):
        pop = np.random.rand(self.population_size, self.dim)
        pop = bounds.lb + (bounds.ub - bounds.lb) * pop
        return pop

    def _mutate(self, target_idx, population):
        selected_indices = np.random.choice(np.delete(np.arange(self.population_size), target_idx), 3, replace=False)
        a, b, c = population[selected_indices]
        mutant = a + self.mutation_factor * (b - c)
        return mutant

    def _crossover(self, target, mutant, bounds):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return np.clip(trial, bounds.lb, bounds.ub)
    
    def _evaluate_population(self, func, population):
        scores = np.array([func(ind) for ind in population])
        self.current_budget += len(population)
        return scores

    def _local_optimization(self, best, func, bounds):
        result = minimize(func, best, method='L-BFGS-B', bounds=list(zip(bounds.lb, bounds.ub)))
        return result.x if result.success else best

    def _enforce_periodicity(self, solution):
        period = self.dim // 2
        for i in range(period):
            solution[i + period] = solution[i]
        return solution

    def _build_surrogate_model(self, population, scores):
        self.surrogate_model.fit(population, scores)

    def _predict_surrogate(self, candidate):
        return self.surrogate_model.predict(candidate.reshape(1, -1))[0]

    def __call__(self, func):
        bounds = func.bounds
        population = self._initialize_population(bounds)
        scores = self._evaluate_population(func, population)

        while self.current_budget < self.budget:
            self._build_surrogate_model(population, scores)
            for i in range(self.population_size):
                if self.current_budget >= self.budget:
                    break
                target = population[i]
                mutant = self._mutate(i, population)
                var_pred = np.var(self.surrogate_model.predict(population))
                self.mutation_factor = 0.5 + 0.4 * var_pred
                self.crossover_rate = 0.7 + 0.2 * np.random.rand()
                trial = self._crossover(target, mutant, bounds)
                trial = self._enforce_periodicity(trial)
                surrogate_score = self._predict_surrogate(trial)
                trial_score = surrogate_score + (func(trial) - surrogate_score) if np.random.rand() < 0.3 else surrogate_score
                self.current_budget += 1
                if trial_score < scores[i]:
                    population[i] = trial
                    scores[i] = trial_score

            best, best_score = population[np.argmin(scores)], np.min(scores)
            best = self._local_optimization(best, func, bounds)
            best = self._enforce_periodicity(best)

        return best