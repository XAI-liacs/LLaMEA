import numpy as np

class MemeticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 10 * dim  # A reasonable population size
        self.CR = 0.9  # Crossover probability
        self.F = 0.8   # Differential weight
        self.local_search_prob = 0.2  # Probability of conducting local search

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                
                # Mutation
                indices = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                a, b, c = pop[indices]
                mutant = np.clip(a + self.F * (b - c), *self.bounds)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    pop[i], fitness[i] = trial, trial_fitness

                # Local search occasionally
                if np.random.rand() < self.local_search_prob:
                    trial = self.local_search(trial, func)
                    trial_fitness = func(trial)
                    eval_count += 1
                    if trial_fitness < fitness[i]:
                        pop[i], fitness[i] = trial, trial_fitness

        return pop[np.argmin(fitness)]

    def local_search(self, solution, func):
        best = solution
        best_fitness = func(best)
        step_size = 0.1  # Small step size for local search
        for i in range(self.dim):
            for direction in [-1, 1]:
                candidate = np.clip(best.copy(), *self.bounds)
                candidate[i] += direction * step_size
                candidate_fitness = func(candidate)
                if candidate_fitness < best_fitness:
                    best = candidate
                    best_fitness = candidate_fitness
        return best