import numpy as np

class ADEQMRM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, float('inf'))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.F = 0.5
        self.CR = 0.9
        self.beta = 0.3
        self.evaluations = 0

    def quantum_rotational_mutation(self, base, target, scale=0.01):
        angle = np.random.normal(0, 1) * scale
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        idx = np.random.choice(self.dim, 2, replace=False)
        temp_vec = np.array([base[idx[0]], base[idx[1]]]) - np.array([target[idx[0]], target[idx[1]]])
        rotated_vec = np.dot(rotation_matrix, temp_vec)
        base[idx[0]] += rotated_vec[0]
        base[idx[1]] += rotated_vec[1]
        return base

    def mutate(self, idx):
        candidates = np.random.choice([i for i in range(self.population_size) if i != idx], 3, replace=False)
        base, a, b = self.positions[candidates]
        mutant = base + self.F * (a - b)
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        trial = np.where(cross_points, mutant, target)
        return trial

    def select(self, target_idx, trial, trial_fitness):
        if trial_fitness < self.fitness[target_idx]:
            self.positions[target_idx] = trial
            self.fitness[target_idx] = trial_fitness
            if trial_fitness < self.best_fitness:
                self.best_solution = trial
                self.best_fitness = trial_fitness

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            self.fitness[i] = func(self.positions[i])
            if self.fitness[i] < self.best_fitness:
                self.best_solution = self.positions[i]
                self.best_fitness = self.fitness[i]
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.best_solution

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.positions[i], mutant)
                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
                trial = self.quantum_rotational_mutation(trial, self.positions[i])
                trial_fitness = func(trial)
                self.select(i, trial, trial_fitness)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

        return self.best_solution