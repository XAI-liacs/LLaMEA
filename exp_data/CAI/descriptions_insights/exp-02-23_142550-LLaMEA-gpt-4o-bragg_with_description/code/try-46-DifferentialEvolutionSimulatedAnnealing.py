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
        diversity_factor = np.std(self.population) / 10.0  # New diversity factor
        return self.population[a] + F_adaptive * (self.population[b] - self.population[c]) * exploration_factor + diversity_factor

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_prob
        offspring = np.where(crossover_mask, mutant, target)
        return offspring

    def select(self, target, trial, func):
        target_score = func(target)
        trial_score = func(trial)
        self.evaluations += 2
        decay_factor = (self.budget - self.evaluations) / self.budget  # New decay factor
        acceptance_prob = np.exp((target_score - trial_score) / (self.temperature * decay_factor))  # Modified acceptance probability
        if trial_score < target_score or np.random.rand() < acceptance_prob:
            return trial, trial_score
        return target, target_score

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        
        while self.evaluations < self.budget:
            new_population = np.zeros_like(self.population)
            for i in range(self.pop_size):
                mutant = self.mutate(i)
                mutant = np.clip(mutant, lb, ub)
                trial = self.crossover(self.population[i], mutant)
                trial = np.clip(trial, lb, ub)
                new_population[i], score = self.select(self.population[i], trial, func)
                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = new_population[i]
            elitism_idx = np.argmin([func(ind) for ind in new_population])
            self.population = new_population
            self.population[elitism_idx] = self.best_solution
            self.cooling_rate = 0.95 + 0.05 * (self.evaluations / self.budget)  # Adjust cooling rate adaptively
            
        return self.best_solution, self.best_score