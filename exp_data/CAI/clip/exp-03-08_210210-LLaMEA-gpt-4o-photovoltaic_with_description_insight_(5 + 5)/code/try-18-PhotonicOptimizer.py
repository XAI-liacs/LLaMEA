import numpy as np

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.global_pop_size = 50
        self.local_refinement_steps = 10
        self.dim_increment = max(dim // 10, 1)
        self.robustness_factor = 0.01

    def differential_evolution(self, func, bounds, pop_size, max_iter):
        lb, ub = bounds.lb, bounds.ub
        pop = lb + (ub - lb) * np.random.rand(pop_size, len(lb))
        fitness = np.apply_along_axis(func, 1, pop)
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate

        for iteration in range(max_iter):
            if len(fitness) >= self.budget:
                break

            for i in range(pop_size):
                indices = list(range(pop_size))
                indices.remove(i)
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                F = 0.5 + 0.3 * np.random.rand()  # Self-adaptive mutation factor
                mutant = np.clip(a + F * (b - c), lb, ub)
                CR = 0.8 + 0.2 * np.random.rand()  # Self-adaptive crossover rate
                trial = np.where(np.random.rand(len(lb)) < CR, mutant, pop[i])
                trial_fitness = func(trial)

                if len(fitness) >= self.budget:
                    break

                if trial_fitness < fitness[i]:
                    pop[i], fitness[i] = trial, trial_fitness

        return pop, fitness

    def local_search(self, func, solution, bounds):
        best_solution = solution
        best_fitness = func(solution)
        step_size = (bounds.ub - bounds.lb) / 100.0
        perturbation_scale = 0.05

        for _ in range(self.local_refinement_steps):
            neighbors = [solution + step_size * np.random.randn(self.dim) for _ in range(5)]
            for neighbor in neighbors:
                neighbor = np.clip(neighbor, bounds.lb, bounds.ub)  # Ensure within bounds
                neighbor_fitness = func(neighbor)
                if neighbor_fitness < best_fitness:
                    best_solution, best_fitness = neighbor, neighbor_fitness
                    step_size *= 0.5  # Adaptive step size reduction
                # Perturbation analysis for robustness
                if np.random.rand() < perturbation_scale:
                    test_perturbation = neighbor + perturbation_scale * np.random.randn(self.dim)
                    test_perturbation = np.clip(test_perturbation, bounds.lb, bounds.ub)
                    test_fitness = func(test_perturbation)
                    if test_fitness < best_fitness:
                        best_solution, best_fitness = test_perturbation, test_fitness
        return best_solution, best_fitness

    def __call__(self, func):
        bounds = func.bounds
        current_dim = self.dim_increment
        best_solution = None
        best_fitness = float('inf')

        while current_dim <= self.dim:
            pop, fitness = self.differential_evolution(func, bounds, self.global_pop_size, self.budget // 10)
            for solution in pop:
                if len(fitness) >= self.budget:
                    break
                refined_solution, refined_fitness = self.local_search(func, solution, bounds)
                if refined_fitness < best_fitness:
                    best_solution, best_fitness = refined_solution, refined_fitness

            current_dim += self.dim_increment

        return best_solution