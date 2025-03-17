import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10 * self.dim
        F = 0.5  # Differential weight
        CR = 0.9  # Crossover probability

        # Initialize population
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        func_evals = population_size

        # Store the best solution found
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        while func_evals < self.budget:
            # Dynamic population size adjustment
            population_size = max(5, population_size - 1)

            for i in range(population_size):
                # Mutation: Create a mutant vector
                indices = np.random.choice(np.delete(np.arange(population_size), i), 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + F * (b - c), lb, ub)

                # Adaptive crossover probability
                CR = 0.8 + 0.2 * np.random.rand()

                # Crossover: Create a trial vector
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[i])

                # Selection: Evaluate and select the better solution
                trial_fitness = func(trial)
                func_evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Update the best solution found
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

                if func_evals >= self.budget:
                    break

            # Local search around the best solution to refine
            perturbation_scale = 0.05 * (1 - func_evals / self.budget)  # Decrease perturbation over time
            perturbation = np.random.normal(0, perturbation_scale, self.dim)  # Variable perturbation
            candidate = np.clip(best_solution + perturbation, lb, ub)
            candidate_fitness = func(candidate)
            func_evals += 1
            if candidate_fitness < best_fitness:
                best_solution = candidate
                best_fitness = candidate_fitness

        return best_solution