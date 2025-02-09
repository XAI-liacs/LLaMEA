import numpy as np

class AdaptiveMemeticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 5 * dim
        self.min_population_size = 2 * dim
        self.max_population_size = 10 * dim
        self.crossover_rate = 0.9
        self.mutation_factor = 0.8
        self.local_search_prob = 0.2
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        while self.evaluations < self.budget:
            for i in range(population_size):
                # Mutation
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update best solution
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                # Local search
                if np.random.rand() < self.local_search_prob and self.evaluations < self.budget:
                    improved, trial_fitness = self.local_search(trial, func, lb, ub)
                    if improved:
                        population[i] = trial
                        fitness[i] = trial_fitness
                        if trial_fitness < best_fitness:
                            best_solution = trial
                            best_fitness = trial_fitness
            
            # Dynamic Population Sizing
            if self.evaluations < self.budget / 2:
                population_size = min(self.max_population_size, population_size + dim)
            else:
                population_size = max(self.min_population_size, population_size - dim)
            if self.evaluations < self.budget:
                additional_population = np.random.uniform(lb, ub, (population_size - len(population), self.dim))
                population = np.vstack((population, additional_population))
                fitness = np.append(fitness, [func(ind) for ind in additional_population])
                self.evaluations += len(additional_population)

        return best_solution

    def local_search(self, solution, func, lb, ub):
        improved = False
        step_size = (ub - lb) * 0.01
        current_fitness = func(solution)
        for _ in range(5):
            candidate = solution + np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(candidate, lb, ub)
            candidate_fitness = func(candidate)
            self.evaluations += 1
            if candidate_fitness < current_fitness:
                solution = candidate
                current_fitness = candidate_fitness
                improved = True
        return improved, current_fitness