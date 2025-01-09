import numpy as np

class HybridAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Set a reasonable population size
        self.cr = 0.9  # Crossover probability
        self.f = 0.8  # Differential weight
        self.lb = -5.0  # Lower bound
        self.ub = 5.0  # Upper bound
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def local_search(self, individual, func):
        # A simple local search strategy to refine the solution
        step_size = 0.01
        for _ in range(5):  # Perform few local moves
            candidate = individual + np.random.uniform(-step_size, step_size, size=self.dim)
            candidate = np.clip(candidate, self.lb, self.ub)
            candidate_fitness = func(candidate)
            if candidate_fitness < self.best_fitness:
                self.best_solution = candidate
                self.best_fitness = candidate_fitness
                individual = candidate  # Update the individual
        return individual

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                # Mutation
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[idxs]
                mutant = a + self.f * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, self.population[i])

                # Evaluation
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < func(self.population[i]):
                    self.population[i] = trial
                    if trial_fitness < self.best_fitness:
                        self.best_solution = trial
                        self.best_fitness = trial_fitness

                # Apply local search to improve exploration
                if evaluations % (self.population_size // 2) == 0:
                    self.population[i] = self.local_search(self.population[i], func)
                    evaluations += 1  # Each local search step counts as an evaluation

        return self.best_solution