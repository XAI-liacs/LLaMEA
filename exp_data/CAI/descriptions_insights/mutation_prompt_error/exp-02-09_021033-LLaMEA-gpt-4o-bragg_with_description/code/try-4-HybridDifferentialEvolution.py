import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.adaptive_local_search_rate = 0.2
        self.bounds = None
        
    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                self.mutation_factor = 0.6 + 0.4 * (evaluations / self.budget)  # Line changed
                mutant = np.clip(a + self.mutation_factor * (b - c), self.bounds[0], self.bounds[1])
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                self.crossover_rate = 0.6 + 0.3 * np.random.rand()  # Line changed
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if np.random.rand() < self.adaptive_local_search_rate:
                    local_trial = self.local_search(trial, func, evaluations)
                    evaluations += 1
                    local_trial_fitness = func(local_trial)
                    if local_trial_fitness < fitness[i]:
                        population[i] = local_trial
                        fitness[i] = local_trial_fitness
                        if local_trial_fitness < best_fitness:
                            best_solution = local_trial
                            best_fitness = local_trial_fitness

        return best_solution
    
    def local_search(self, solution, func, evaluations):
        local = solution.copy()
        adaptive_perturbation = 0.1 * (1 - evaluations / self.budget)  # Adaptive scaling
        perturbation = np.random.normal(0, adaptive_perturbation, self.dim)  # Line changed
        local += perturbation
        local = np.clip(local, self.bounds[0], self.bounds[1])
        return local