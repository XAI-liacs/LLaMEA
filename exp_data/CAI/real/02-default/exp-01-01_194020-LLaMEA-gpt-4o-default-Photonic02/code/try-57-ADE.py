import numpy as np

class ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.scaling_factor = 0.5
        self.crossover_rate = 0.7
        self.population = []
        self.best_solution = None
        self.best_value = float('inf')

    def initialize_population(self, lb, ub):
        return [lb + (ub - lb) * np.random.rand(self.dim) for _ in range(self.pop_size)]

    def mutate_and_crossover(self, target_idx, lb, ub):
        idxs = [idx for idx in range(self.pop_size) if idx != target_idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = self.population[a] + self.scaling_factor * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, lb, ub)
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, self.population[target_idx])
        return trial

    def chaotic_local_search(self, candidate, lb, ub, iterations=10):
        chaotic_sequence = np.sin(np.array(range(iterations)) * np.pi * 0.1)
        for i in range(iterations):
            perturbation = chaotic_sequence[i] * (ub - lb) * 0.01
            candidate = np.clip(candidate + perturbation, lb, ub)
        return candidate

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.population = self.initialize_population(lb, ub)
        
        while evaluations < self.budget:
            for target_idx, target in enumerate(self.population):
                trial = self.mutate_and_crossover(target_idx, lb, ub)
                trial_value = func(trial)
                evaluations += 1

                if trial_value < self.best_value:
                    self.best_value = trial_value
                    self.best_solution = trial.copy()

                if trial_value < func(target):
                    self.population[target_idx] = trial
                elif np.random.rand() < 0.1:
                    candidate = self.chaotic_local_search(target, lb, ub)
                    candidate_value = func(candidate)
                    evaluations += 1
                    if candidate_value < self.best_value:
                        self.best_value = candidate_value
                        self.best_solution = candidate.copy()

                if evaluations >= self.budget:
                    break

            self.scaling_factor = 0.5 + 0.5 * (1 - evaluations / self.budget)
            self.crossover_rate = 0.7 + 0.3 * np.sin(evaluations * np.pi / (2 * self.budget))

        return self.best_solution, self.best_value