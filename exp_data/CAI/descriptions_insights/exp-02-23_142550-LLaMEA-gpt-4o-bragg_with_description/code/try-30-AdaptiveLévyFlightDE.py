import numpy as np
from scipy.stats import levy

class AdaptiveLévyFlightDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30  # Number of individuals
        self.mutation_factor = 0.8
        self.crossover_prob = 0.7
        self.temperature = 100.0
        self.cooling_rate = 0.95
        self.population = np.random.uniform(-1.0, 1.0, (self.pop_size, self.dim))
        self.best_solution = None
        self.best_score = np.inf
        self.evaluations = 0

    def mutate(self, idx, diversity):
        indices = [i for i in range(self.pop_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        F_adaptive = self.mutation_factor * (1 - self.evaluations / self.budget) * (1 + np.std(diversity))
        exploration_factor = np.random.uniform(0.5, 1.5)
        return self.population[a] + F_adaptive * (self.population[b] - self.population[c]) * exploration_factor

    def levy_flight(self, dim):
        step = levy.rvs(size=dim)
        return step

    def crossover(self, target, mutant, diversity):
        cross_prob = self.crossover_prob + np.std(diversity) * 0.2
        crossover_mask = np.random.rand(self.dim) < cross_prob
        offspring = np.where(crossover_mask, mutant, target)
        return offspring

    def select(self, target, trial, func):
        target_score = func(target)
        trial_score = func(trial)
        self.evaluations += 2
        if trial_score < target_score or np.random.rand() < np.exp((target_score - trial_score) / self.temperature):
            return trial, trial_score
        return target, target_score

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        
        while self.evaluations < self.budget:
            new_population = np.zeros_like(self.population)
            diversity = np.std(self.population, axis=0)
            
            for i in range(self.pop_size):
                if np.random.rand() < 0.3:  # Probability for Lévy flight
                    mutant = self.population[i] + self.levy_flight(self.dim)
                else:
                    mutant = self.mutate(i, diversity)
                
                mutant = np.clip(mutant, lb, ub)
                trial = self.crossover(self.population[i], mutant, diversity)
                trial = np.clip(trial, lb, ub)
                new_population[i], score = self.select(self.population[i], trial, func)
                
                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = new_population[i]
            
            elitism_idx = np.argmin([func(ind) for ind in new_population])
            self.population = new_population
            self.population[elitism_idx] = self.best_solution
            self.temperature *= self.cooling_rate
            
        return self.best_solution, self.best_score