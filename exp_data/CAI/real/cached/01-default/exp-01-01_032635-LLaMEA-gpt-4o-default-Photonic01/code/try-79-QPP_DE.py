import numpy as np

class QPP_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget)
        self.population = None
        self.best_solution = None
        self.best_value = np.inf
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.crowding_factor = 0.1  # Crowding factor for local search
        self.quantum_perturbation_rate = 0.2

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.bounds = (lb, ub)

    def differential_mutation(self, idx):
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
        lb, ub = self.bounds
        return np.clip(mutant, lb, ub)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def quantum_perturbation(self, solution, global_best):
        if np.random.rand() < self.quantum_perturbation_rate:
            beta = np.random.normal(0, 1, self.dim)
            delta = np.random.normal(0, 1, self.dim) * 0.05
            perturbed = solution + beta * (global_best - solution) + delta
            lb, ub = self.bounds
            return np.clip(perturbed, lb, ub)
        return solution

    def crowding_local_search(self):
        for idx in np.argsort([np.linalg.norm(ind - self.best_solution) for ind in self.population])[:int(self.crowding_factor * self.population_size)]:
            perturbation = np.random.uniform(-0.1, 0.1, self.dim)
            self.population[idx] = np.clip(self.population[idx] + perturbation, *self.bounds)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                mutant = self.differential_mutation(i)
                trial = self.crossover(self.population[i], mutant)

                trial = self.quantum_perturbation(trial, self.best_solution if self.best_solution is not None else trial)

                trial_value = func(trial)
                evaluations += 1

                if trial_value < func(self.population[i]):
                    self.population[i] = trial
                    if trial_value < self.best_value:
                        self.best_value = trial_value
                        self.best_solution = trial

            self.crowding_local_search()

        return self.best_solution, self.best_value