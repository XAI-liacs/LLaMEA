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
        CR = 0.9  # Initial crossover probability

        # Initialize population
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        func_evals = population_size

        # Store the best solution found
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        while func_evals < self.budget:
            pop_convergence_factor = 1 - (best_fitness / np.max(fitness))
            population_size = max(5, int(10 * self.dim * pop_convergence_factor))
            
            for i in range(population_size):
                tournament_indices = np.random.choice(population_size, 5, replace=False)
                tournament_fitness = fitness[tournament_indices]
                selected_indices = tournament_indices[np.argsort(tournament_fitness)[:3]]
                a, b, c = population[selected_indices]

                # Lévy flight-inspired mutation
                levy = np.random.standard_normal(self.dim) * np.power(np.random.uniform(0, 1), -1.0 / 3.0)
                mutant = np.clip(a + F * (b - c) + levy, lb, ub)

                # Adaptive Crossover: Adjust based on fitness diversity
                fitness_diversity = np.std(fitness) / (np.mean(fitness) + 1e-8)
                CR = 0.9 * (1 - fitness_diversity)

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

            perturbation = np.random.normal(0, 0.1, self.dim)
            candidate = np.clip(best_solution + perturbation, lb, ub)
            candidate_fitness = func(candidate)
            func_evals += 1
            if candidate_fitness < best_fitness:
                best_solution = candidate
                best_fitness = candidate_fitness

        return best_solution