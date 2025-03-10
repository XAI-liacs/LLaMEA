import numpy as np

class HybridMetaheuristicRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.current_evals = 0

    def adaptive_differential_evolution(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        fitness = np.array([func(ind) for ind in population])
        self.current_evals += self.population_size
        
        while self.current_evals < self.budget:
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                f_trial = func(trial)
                self.current_evals += 1

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                elif np.random.rand() < 0.1:
                    self.mutation_factor = np.random.uniform(0.6, 1.0)
                    self.crossover_rate = np.random.uniform(0.5, 0.9)

                if self.current_evals >= self.budget:
                    break

        return population[np.argmin(fitness)]

    def enhanced_local_search(self, best_solution, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        step_size = 0.1 * (ub - lb)
        
        for _ in range(20):  # increased iterations for local search
            for d in range(self.dim):
                perturbed_solution = np.copy(best_solution)
                perturbed_solution[d] += step_size[d] * np.random.randn()
                perturbed_solution = np.clip(perturbed_solution, lb, ub)
                f_perturbed = func(perturbed_solution)
                self.current_evals += 1

                if f_perturbed < func(best_solution):
                    best_solution = perturbed_solution

                if self.current_evals >= self.budget:
                    break

        return best_solution
    
    def __call__(self, func):
        best_solution = self.adaptive_differential_evolution(func)
        best_solution = self.enhanced_local_search(best_solution, func)
        return best_solution