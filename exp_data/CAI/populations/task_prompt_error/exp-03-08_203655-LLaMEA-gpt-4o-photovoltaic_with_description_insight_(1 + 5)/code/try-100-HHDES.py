import numpy as np

class HHDES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.CR = 0.9
        self.F = 0.8
        self.pbest_fraction = 0.2
        self.population = np.random.rand(self.pop_size, dim)
        self.best_solution = np.random.rand(dim)  # Initialization fix
        self.best_fitness = np.inf
        self.evaluate_budget = 0
        self.layer_increase = 1

    def evaluate(self, func, candidate):
        self.evaluate_budget += 1
        return func(candidate)

    def differential_evolution(self, func):
        for _ in range(self.budget):
            if self.evaluate_budget >= self.budget:
                break
            fitnesses = [self.evaluate(func, ind) for ind in self.population]
            best_local_idx = np.argmin(fitnesses)
            if fitnesses[best_local_idx] < self.best_fitness:
                self.best_fitness = fitnesses[best_local_idx]
                self.best_solution = self.population[best_local_idx]
            mean_fitness = np.mean(fitnesses)
            fitness_diversity = np.std(fitnesses) / (mean_fitness + 1e-9)
            if fitness_diversity < 0.1:
                self.layer_increase = min(self.layer_increase + 1, self.dim)
            else:
                self.layer_increase = max(self.layer_increase - 1, 1)
            for i in range(self.pop_size):
                if self.evaluate_budget >= self.budget:
                    break
                donor = self.adaptive_mutate(i)  # Changed to adaptive mutation
                trial = self.adaptive_crossover(self.population[i], donor)  # Adaptive crossover
                trial_fitness = self.evaluate(func, trial)
                if trial_fitness < fitnesses[i]:
                    self.population[i] = trial

    def adaptive_mutate(self, current_idx):
        indices = list(range(self.pop_size))
        indices.remove(current_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        F = 0.5 + np.random.rand() * 0.3  # Adaptive mutation factor
        mutant = self.population[a] + F * (self.population[b] - self.population[c])
        return np.clip(mutant, 0, 1)

    def adaptive_crossover(self, target, donor):
        CR = 0.8 + np.random.rand() * 0.2  # Adaptive crossover rate
        cross_points = np.random.rand(self.dim) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, donor, target)

    def local_search(self, func, solution):
        perturbation = np.random.normal(0, 0.1, solution.shape)
        candidate = solution + perturbation
        candidate = np.clip(candidate, 0, 1)
        candidate_fitness = self.evaluate(func, candidate)
        if candidate_fitness < self.evaluate(func, solution):
            return candidate, candidate_fitness
        return solution, self.evaluate(func, solution)

    def optimize_layers(self, func):
        for _ in range(self.budget // self.dim):
            if self.evaluate_budget >= self.budget:
                break
            self.differential_evolution(func)
            self.best_solution, self.best_fitness = self.local_search(func, self.best_solution)

    def __call__(self, func):
        self.optimize_layers(func)
        return self.best_solution