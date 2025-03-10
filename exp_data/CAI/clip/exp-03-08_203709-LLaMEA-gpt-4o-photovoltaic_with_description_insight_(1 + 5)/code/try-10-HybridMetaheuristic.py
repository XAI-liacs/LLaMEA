import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size for DE
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = None

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def mutate(self, idx, candidates, bounds):
        a, b, c = self.population[np.random.choice(candidates, 3, replace=False)]
        adaptive_F = self.F * np.random.uniform(0.8, 1.2)  # Adaptive mutation
        mutant = a + adaptive_F * (b - c)
        return np.clip(mutant, bounds.lb, bounds.ub)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def adaptive_local_refinement(self, candidate, func, bounds, convergence_speed):
        perturbation_strength = max(0.05 * convergence_speed, 0.01)
        for _ in range(5):  # small local search steps
            perturbation = np.random.normal(0, perturbation_strength, self.dim)  # Normal perturbation
            trial = np.clip(candidate + perturbation, bounds.lb, bounds.ub)
            if func(trial) > func(candidate):
                candidate = trial
                break
        return candidate

    def __call__(self, func):
        bounds = func.bounds
        self.population = self.initialize_population(bounds)
        fitness = np.array([func(ind) for ind in self.population])
        num_evaluations = self.pop_size
        prev_best_fitness = np.max(fitness)

        while num_evaluations < self.budget:
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