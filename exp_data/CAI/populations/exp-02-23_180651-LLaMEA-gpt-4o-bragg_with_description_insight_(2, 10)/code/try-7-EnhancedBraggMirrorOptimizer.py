import numpy as np

class EnhancedBraggMirrorOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # DE mutation factor
        self.CR = 0.9  # DE crossover probability
        self.population = None
        self.best_solution = None
        self.best_fitness = float('-inf')

    def quasi_oppositional_init(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        opposite_population = lb + ub - self.population
        self.population = np.vstack((self.population, opposite_population))
        self.population_size = self.population.shape[0]

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_solution = self.population[best_idx]
        return fitness

    def mutate(self, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        return self.population[a] + self.F * (self.population[b] - self.population[c])

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def adaptive_parameters(self, eval_ratio):
        self.F = 0.4 + 0.1 * np.sin(np.pi * eval_ratio)
        self.CR = 0.8 + 0.1 * np.cos(np.pi * eval_ratio)

    def periodic_cost_function(self, candidate):
        periodic_penalty = np.var(np.diff(candidate.reshape(-1, 2), axis=0))
        return periodic_penalty

    def stochastic_ranking(self, fitness, penalties, tau=0.45):
        indices = np.arange(len(fitness))
        scores = fitness - penalties
        for i in range(len(scores) - 1):
            for j in range(len(scores) - 1 - i):
                if np.random.random() < tau:
                    if scores[j] < scores[j + 1]:
                        indices[j], indices[j + 1] = indices[j + 1], indices[j]
                else:
                    if penalties[j] > penalties[j + 1]:
                        indices[j], indices[j + 1] = indices[j + 1], indices[j]
        return indices

    def optimize(self, func):
        evaluations = 0
        while evaluations < self.budget:
            self.adaptive_parameters(evaluations / self.budget)
            fitness = self.evaluate_population(func)
            penalties = np.array([self.periodic_cost_function(ind) for ind in self.population])
            ranked_indices = self.stochastic_ranking(fitness, penalties)
            new_population = np.empty_like(self.population)
            for i in ranked_indices:
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                
                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
                trial_fitness = func(trial) - self.periodic_cost_function(trial)
                
                if trial_fitness > fitness[i]:
                    new_population[i] = trial
                else:
                    new_population[i] = self.population[i]
                evaluations += 1
                if evaluations >= self.budget:
                    break

            self.population = new_population

        return self.best_solution

    def __call__(self, func):
        self.quasi_oppositional_init(func.bounds.lb, func.bounds.ub)
        return self.optimize(func)