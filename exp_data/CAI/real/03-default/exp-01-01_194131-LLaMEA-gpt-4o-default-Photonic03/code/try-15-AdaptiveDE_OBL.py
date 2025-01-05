import numpy as np

class AdaptiveDE_OBL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, float('inf'))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.evaluations = 0
        self.F = 0.5  # Scaling factor
        self.CR = 0.9  # Crossover probability
        self.mutation_strategies = [
            self.de_rand_1,
            self.de_best_1,
            self.de_current_to_best_1
        ]

    def obl(self, bounds):
        # Opposition-based Learning for initial population
        return bounds.lb + bounds.ub - self.positions

    def de_rand_1(self, target_idx):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        a, b, c = self.positions[idxs]
        return a + self.F * (b - c)

    def de_best_1(self, target_idx):
        idxs = np.random.choice(self.population_size, 2, replace=False)
        a, b = self.positions[idxs]
        return self.best_solution + self.F * (a - b)

    def de_current_to_best_1(self, target_idx):
        idxs = np.random.choice(self.population_size, 2, replace=False)
        a, b = self.positions[idxs]
        return self.positions[target_idx] + self.F * (self.best_solution - self.positions[target_idx]) + self.F * (a - b)

    def mutate(self, idx, func):
        strategy = np.random.choice(self.mutation_strategies)
        mutant = strategy(idx)
        for i in range(self.dim):
            if np.random.rand() > self.CR:
                mutant[i] = self.positions[idx, i]
        mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
        return mutant

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        opposite_positions = self.obl(func.bounds)
        
        # Combine initial and opposite populations
        all_positions = np.vstack((self.positions, opposite_positions))
        all_fitness = np.array([func(pos) for pos in all_positions])
        
        # Select best individuals
        best_idxs = np.argsort(all_fitness)[:self.population_size]
        self.positions = all_positions[best_idxs]
        self.fitness = all_fitness[best_idxs]
        
        self.best_solution = self.positions[0]
        self.best_fitness = self.fitness[0]
        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                trial = self.mutate(i, func)
                trial_fitness = func(trial)
                
                if trial_fitness < self.fitness[i]:
                    self.positions[i] = trial
                    self.fitness[i] = trial_fitness

                if trial_fitness < self.best_fitness:
                    self.best_solution = trial
                    self.best_fitness = trial_fitness
                
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

        return self.best_solution