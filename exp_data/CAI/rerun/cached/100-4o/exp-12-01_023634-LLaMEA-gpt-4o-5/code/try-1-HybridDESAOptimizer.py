import numpy as np

class HybridDESAOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20 + self.dim
        self.F = 0.9  # Differential weight increased for diversity
        self.CR = 0.9  # Crossover probability
        self.T0 = 100  # Initial temperature for SA
        self.alpha = 0.98  # Adjusted cooling rate for faster convergence
        self.lb = -5.0
        self.ub = 5.0
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.current_budget = 0

    def __call__(self, func):
        fitness = np.apply_along_axis(func, 1, self.population)
        self.current_budget += self.pop_size

        best_idx = np.argmin(fitness)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = fitness[best_idx]

        while self.current_budget < self.budget:
            new_population = np.zeros_like(self.population)
            temperature = self.T0

            for i in range(self.pop_size):
                # Differential Evolution mutation
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lb, self.ub)

                # Adaptive Crossover
                self.CR = 0.7 + 0.3 * (self.best_fitness / (self.best_fitness + fitness[i]))
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, self.population[i])

                # Apply Simulated Annealing acceptance criterion
                trial_fitness = func(trial)
                self.current_budget += 1

                if trial_fitness < fitness[i]:
                    new_population[i], fitness[i] = trial, trial_fitness
                else:
                    delta = trial_fitness - fitness[i]
                    if np.random.rand() < np.exp(-delta / temperature):
                        new_population[i], fitness[i] = trial, trial_fitness

                # Update global best
                if fitness[i] < self.best_fitness:
                    self.best_solution = new_population[i].copy()
                    self.best_fitness = fitness[i]

                # Cooling
                temperature *= self.alpha

                if self.current_budget >= self.budget:
                    break

            self.population = new_population

        return self.best_solution