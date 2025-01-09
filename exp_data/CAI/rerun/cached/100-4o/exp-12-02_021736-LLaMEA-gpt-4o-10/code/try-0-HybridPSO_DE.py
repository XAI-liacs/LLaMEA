import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.c1 = 1.49  # cognitive component
        self.c2 = 1.49  # social component
        self.w = 0.72  # inertia weight
        self.f = 0.8  # DE scaling factor
        self.cr = 0.9  # DE crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = self.population.copy()
        self.global_best = self.population[np.argmin([float('inf')] * self.population_size)]
        self.personal_best_scores = np.full(self.population_size, float('inf'))

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            # Evaluate current population
            fitness = np.array([func(ind) for ind in self.population])
            evaluations += self.population_size
            
            # Update personal and global bests
            for i in range(self.population_size):
                if fitness[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness[i]
                    self.personal_best[i] = self.population[i]
            self.global_best = self.population[np.argmin(fitness)]

            # PSO - Update velocities and positions
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            for i in range(self.population_size):
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.personal_best[i] - self.population[i]) +
                                      self.c2 * r2 * (self.global_best - self.population[i]))
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

            # DE - Create new agents using DE strategy
            new_population = self.population.copy()
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])
                if func(trial) < func(self.population[i]):
                    new_population[i] = trial
                evaluations += 1
                if evaluations >= self.budget:
                    break
            self.population = new_population
        
        return self.global_best