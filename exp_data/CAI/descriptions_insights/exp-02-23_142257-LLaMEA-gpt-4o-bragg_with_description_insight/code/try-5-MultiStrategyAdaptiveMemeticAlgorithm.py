import numpy as np
from scipy.optimize import minimize

class MultiStrategyAdaptiveMemeticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Effective population size
        self.subcomponents = 2  # Divide problem into subcomponents
        self.population = None
        self.best_solution = None
        self.best_score = float('-inf')
        self.eval_count = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.best_solution = self.population[0]

    def reflectivity_score(self, x):
        # Reward for solutions with periodic characteristics
        period = self.dim // 2
        periodic_deviation = np.sum((x[:period] - x[period:2*period]) ** 2)
        return periodic_deviation

    def coevolutionary_strategy(self, func, lb, ub):
        # Divide the problem into subcomponents for cooperative coevolution
        sub_size = self.dim // self.subcomponents
        F = np.random.uniform(0.4, 0.8)  # Diverse differential weight
        CR = np.random.uniform(0.7, 0.9)  # Diverse crossover probability
        
        for i in range(self.population_size):
            if self.eval_count >= self.budget:
                break

            idxs = [idx for idx in range(self.population_size) if idx != i]
            for j in range(self.subcomponents):
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.array(self.population[i])
                start = j * sub_size
                end = (j + 1) * sub_size
                mutant[start:end] = np.clip(a[start:end] + F * (b[start:end] - c[start:end]), lb, ub)

                cross_points = np.random.rand(sub_size) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, sub_size)] = True

                trial = np.array(self.population[i])
                trial[start:end] = np.where(cross_points, mutant[start:end], self.population[i][start:end])

                # Selection
                trial_score = func(trial)
                self.eval_count += 1
                penalty = self.reflectivity_score(trial)
                
                if trial_score > func(self.population[i]) - penalty:
                    self.population[i] = trial

                # Update the best solution found
                if trial_score > self.best_score:
                    self.best_score = trial_score
                    self.best_solution = trial

    def local_periodic_search(self, func, lb, ub):
        # Period-preserving local optimization
        result = minimize(func, self.best_solution, bounds=np.c_[lb, ub], method='TNC')
        self.eval_count += result.nfev

        if result.fun > self.best_score:
            self.best_score = result.fun
            self.best_solution = result.x

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)

        while self.eval_count < self.budget:
            self.coevolutionary_strategy(func, lb, ub)
            self.local_periodic_search(func, lb, ub)

        return self.best_solution