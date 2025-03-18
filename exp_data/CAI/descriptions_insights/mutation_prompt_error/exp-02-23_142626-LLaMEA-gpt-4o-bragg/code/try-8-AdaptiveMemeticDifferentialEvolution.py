import numpy as np

class AdaptiveMemeticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.local_search_prob = 0.2  # Probability of local search

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub
        
        # Initialize population
        population = np.random.uniform(lower_bound, upper_bound, 
                                       (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation and crossover
                idxs = np.random.choice(range(self.population_size), 3, replace=False)
                a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                mutant = np.clip(a + self.F * (b - c), lower_bound, upper_bound)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                    
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1
                
                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Local search
                if np.random.rand() < self.local_search_prob:
                    trial = self.local_search(trial, func, lower_bound, upper_bound)
                    trial_fitness = func(trial)
                    evaluations += 1
                    if trial_fitness < fitness[i]:
                        population[i] = trial
                        fitness[i] = trial_fitness

                if evaluations >= self.budget:
                    break

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]

    def local_search(self, individual, func, lb, ub):
        # Simple hill-climbing local search
        step_size = (ub - lb) * 0.05
        for _ in range(10):
            candidate = individual + np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(candidate, lb, ub)
            if func(candidate) < func(individual):
                individual = candidate
        return individual