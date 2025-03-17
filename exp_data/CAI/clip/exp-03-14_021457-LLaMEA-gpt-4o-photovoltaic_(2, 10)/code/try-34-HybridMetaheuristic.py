import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10 * self.dim
        F = 0.5
        CR = 0.9

        # Initialize population
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        func_evals = population_size

        # Store the best solution found
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        while func_evals < self.budget:
            population_size = max(5, int(10 * self.dim * (1 - (best_fitness / np.max(fitness)))))
            for i in range(population_size):
                indices = np.random.choice(np.delete(np.arange(len(population)), i), 3, replace=False)
                a, b, c = population[indices]
                # Change: Dynamic adjustment of F based on median fitness
                F = 0.5 * (1 + (best_fitness - np.median(fitness)) / (np.max(fitness) - np.min(fitness)))
                mutant = np.clip(a + F * (b - c), lb, ub)

                CR = 0.9 * (1 - (func_evals / self.budget))
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[i])

                trial_fitness = func(trial)
                func_evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

                if func_evals >= self.budget:
                    break

            # Local search with Lévy flight
            levy_step = np.random.standard_cauchy(self.dim)
            candidate = np.clip(best_solution + 0.01 * levy_step, lb, ub)
            candidate_fitness = func(candidate)
            func_evals += 1
            if candidate_fitness < best_fitness:
                best_solution = candidate
                best_fitness = candidate_fitness

        return best_solution