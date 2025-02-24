import numpy as np
from scipy.optimize import minimize

class CooperativeCoevolutionDE:
    def __init__(self, budget, dim, subcomponent_size=2):
        self.budget = budget
        self.dim = dim
        self.subcomponent_size = subcomponent_size
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.current_budget = 0

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

    def _evaluate_population(self, func, population, sub_idx):
        scores = np.zeros(self.population_size)
        for i, ind in enumerate(population):
            sub_ind = np.copy(ind)
            self.current_budget += 1
            scores[i] = func(sub_ind) 
        return scores

    def _local_optimization(self, target, func, bounds):
        result = minimize(func, target, method='L-BFGS-B', bounds=list(zip(bounds.lb, bounds.ub)))
        return result.x if result.success else target

    def _decompose_problem(self, dim):
        subcomponents = []
        for i in range(0, dim, self.subcomponent_size):
            subcomponents.append((i, min(dim, i + self.subcomponent_size)))
        return subcomponents

    def __call__(self, func):
        bounds = func.bounds
        population = self._initialize_population(bounds)
        subcomponents = self._decompose_problem(self.dim)

        while self.current_budget < self.budget:
            for sub_idx, (start, end) in enumerate(subcomponents):
                sub_population = population[:, start:end]
                scores = self._evaluate_population(func, sub_population, sub_idx)

                new_population = []  
                for i in range(self.population_size):
                    if self.current_budget >= self.budget:
                        break
                    target = sub_population[i]
                    mutant = self._mutate(i, sub_population)
                    trial = self._crossover(target, mutant, bounds)
                    trial_score = func(trial)
                    self.current_budget += 1
                    if trial_score < scores[i]:
                        new_population.append(trial)
                        scores[i] = trial_score
                    else:
                        new_population.append(target)
                
                sub_population[:] = np.array(new_population)
                population[:, start:end] = sub_population

            best_score_idx = np.argmin(scores)
            best_individual = population[best_score_idx]
            target = best_individual
            best_individual = self._local_optimization(target, func, bounds)

        return best_individual