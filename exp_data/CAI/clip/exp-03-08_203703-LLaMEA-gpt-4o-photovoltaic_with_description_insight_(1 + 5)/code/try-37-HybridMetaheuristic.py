import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.current_evals = 0

    def differential_evolution(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        fitness = np.array([func(ind) for ind in population])
        self.current_evals += self.population_size
        modular_structure = np.random.choice([2, 3], size=self.dim)

        while self.current_evals < self.budget:
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                adaptive_mutation_factor = self.mutation_factor * (1.2 - self.current_evals / self.budget)  # Changed line
                mutant = np.clip(a + adaptive_mutation_factor * (b - c), lb, ub)
                adaptive_crossover_rate = self.crossover_rate * (1 + np.random.rand() * 0.5)  # More adaptive
                cross_points = np.random.rand(self.dim) < adaptive_crossover_rate  
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                f_trial = func(trial) + np.random.normal(0, 0.01)  # Robustness
                self.current_evals += 1

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    modular_structure[i % self.dim] = np.random.choice([2, 3])

                if self.current_evals >= self.budget:
                    break

        return population[np.argmin(fitness)]

    def local_search(self, best_solution, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        step_size = 0.1 * (ub - lb)
        scale_factor = 1.0

        for _ in range(5 + self.dim // 10):
            for d in range(self.dim):
                perturbed_solution = np.copy(best_solution)
                perturbed_solution[d] += scale_factor * step_size[d] * np.random.randn()
                perturbed_solution = np.clip(perturbed_solution, lb, ub)
                f_perturbed = func(perturbed_solution)
                self.current_evals += 1

                if f_perturbed < func(best_solution):
                    best_solution = perturbed_solution
                    scale_factor = 1.1

                if self.current_evals >= self.budget:
                    break

        return best_solution
    
    def __call__(self, func):
        self.population_size = max(5, self.population_size // 2)
        best_solution = self.differential_evolution(func)
        best_solution = self.local_search(best_solution, func)
        return best_solution