import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(20 * dim, 100)  # Increased initial population size cap
        self.F = 0.6  # Adjusted differential weight
        self.CR = 0.8  # Adjusted crossover probability
        self.population = None
        self.layerwise_evolution = True  # New strategy flag

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def mutate(self, idx, candidates, bounds):
        a, b, c = self.population[np.random.choice(candidates, 3, replace=False)]
        mutant = a + self.F * (b - c)
        return np.clip(mutant, bounds.lb, bounds.ub)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < np.clip(self.CR + np.random.normal(0, 0.1), 0, 1)
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def adaptive_local_refinement(self, candidate, func, bounds, convergence_speed):
        perturbation_strength = max(0.03 * convergence_speed, 0.005)  # Adjusted
        for _ in range(5):
            perturbation = np.random.uniform(-perturbation_strength, perturbation_strength, self.dim)
            trial = np.clip(candidate + perturbation, bounds.lb, bounds.ub)
            if np.mean([func(trial) for _ in range(2)]) > func(candidate):  # Reduced evaluations
                candidate = trial
                break
        return candidate

    def layer_wise_optimization(self, func, bounds):
        evol_dim = int(self.dim * 0.1)  # Optimize 10% of layers
        indices = np.random.choice(self.dim, evol_dim, replace=False)
        for idx in indices:
            candidates = [i for i in range(self.pop_size)]
            mutant = self.mutate(idx, candidates, bounds)
            trial = self.crossover(self.population[idx], mutant)
            if func(trial) > func(self.population[idx]):
                self.population[idx] = trial

    def __call__(self, func):
        bounds = func.bounds
        self.population = self.initialize_population(bounds)
        fitness = np.array([func(ind) for ind in self.population])
        num_evaluations = self.pop_size
        prev_best_fitness = np.max(fitness)

        while num_evaluations < self.budget:
            if self.layerwise_evolution:
                self.layer_wise_optimization(func, bounds)
            for i in range(self.pop_size):
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                mutant = self.mutate(i, candidates, bounds)
                trial = self.crossover(self.population[i], mutant)
                current_best_fitness = np.max(fitness)
                convergence_speed = (current_best_fitness - prev_best_fitness) / prev_best_fitness
                trial = self.adaptive_local_refinement(trial, func, bounds, convergence_speed)

                trial_fitness = func(trial)
                num_evaluations += 1
                if trial_fitness > fitness[i]:
                    self.population[i] = trial
                    fitness[i] = trial_fitness

                if num_evaluations >= self.budget:
                    break

            prev_best_fitness = current_best_fitness

        best_idx = np.argmax(fitness)
        return self.population[best_idx]