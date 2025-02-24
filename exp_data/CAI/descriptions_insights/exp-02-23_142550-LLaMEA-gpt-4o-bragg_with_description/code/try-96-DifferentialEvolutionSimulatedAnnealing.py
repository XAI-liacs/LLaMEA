import numpy as np

class DifferentialEvolutionSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.mutation_factor = 0.8
        self.crossover_prob = 0.7
        self.temperature = 100.0
        self.cooling_rate = 0.95
        self.population = np.random.uniform(-1.0, 1.0, (self.pop_size, self.dim))
        self.best_solution = None
        self.best_score = np.inf
        self.evaluations = 0

    def mutate(self, idx):
        indices = [i for i in range(self.pop_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        F_adaptive = self.mutation_factor * (1 - self.evaluations / self.budget)
        exploration_factor = np.random.uniform(0.5, 1.5)
        if np.random.rand() < 0.6:
            return self.population[a] + F_adaptive * (self.population[b] - self.population[c]) * exploration_factor
        else:
            d = np.random.choice(indices)  
            return self.population[idx] + F_adaptive * (self.population[a] - self.population[d]) + F_adaptive * (self.best_solution - self.population[idx])

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_prob
        offspring = np.where(crossover_mask, mutant, target)
        return offspring

    def select(self, target, trial, func):
        target_score = func(target)
        trial_score = func(trial)
        self.evaluations += 2
        decay_factor = (self.budget - self.evaluations) / self.budget
        acceptance_prob = np.exp((target_score - trial_score) / (self.temperature * decay_factor))
        if trial_score < target_score or np.random.rand() < acceptance_prob:
            return trial, trial_score
        return target, target_score

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_variance = np.var(self.population)
        self.crossover_prob = 0.5 + 0.3 * (self.evaluations / self.budget) + 0.2 * population_variance
        self.population = np.random.uniform(lb, ub, (self.pop_size, self.dim)) * 0.8
        while self.evaluations < self.budget:
            new_population = np.zeros_like(self.population)
            self.pop_size = int(20 + 15 * (self.evaluations / self.budget))
            for i in range(self.pop_size):
                mutant = self.mutate(i)
                mutant = np.clip(mutant, lb, ub)
                trial = self.crossover(self.population[i], mutant)
                trial = np.clip(trial, lb, ub)
                new_population[i], score = self.select(self.population[i], trial, func)
                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = new_population[i]
            elitism_idx = np.argmin([func(new_population[i]) for i in range(self.pop_size)])
            self.population = new_population
            self.population[elitism_idx] = 0.5 * self.best_solution + 0.5 * self.population[elitism_idx]
            self.cooling_rate = 0.9 + 0.05 * (self.evaluations / self.budget)
            
        return self.best_solution, self.best_score