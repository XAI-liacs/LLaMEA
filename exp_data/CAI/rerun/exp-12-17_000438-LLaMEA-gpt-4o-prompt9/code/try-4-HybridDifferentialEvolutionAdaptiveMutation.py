import numpy as np

class HybridDifferentialEvolutionAdaptiveMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * self.dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.CR = 0.9  # Crossover rate
        self.F = 0.8   # Differential weight
        self.mutation_factor_bounds = (0.5, 1.0)

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            new_population = np.zeros_like(self.population)
            for i in range(self.population_size):
                # Adaptive mutation with dynamic scaling factor
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                F_dynamic = self.F + (np.random.rand() - 0.5) * 0.1
                if np.random.rand() < 0.2:  # Adaptive mutation threshold
                    F_dynamic = np.random.uniform(*self.mutation_factor_bounds)
                mutant = np.clip(a + F_dynamic * (b - c), self.lower_bound, self.upper_bound)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial
                if trial_fitness < func(self.population[i]):
                    new_population[i] = trial
                else:
                    new_population[i] = self.population[i]

                if evaluations >= self.budget:
                    break

            # Adaptive Metropolis Sampling for exploration
            if evaluations < self.budget:
                current_cov = np.cov(new_population.T) * 2.4**2 / self.dim
                proposal = np.random.multivariate_normal(self.best_solution, current_cov)
                proposal = np.clip(proposal, self.lower_bound, self.upper_bound)
                proposal_fitness = func(proposal)
                evaluations += 1
                if proposal_fitness < self.best_fitness:
                    self.best_fitness = proposal_fitness
                    self.best_solution = proposal

            if evaluations >= self.budget:
                break

            # Update population with new solutions
            self.population = new_population

        return self.best_solution, self.best_fitness