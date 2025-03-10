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

        for iteration in range(max_iter):
            if len(fitness) >= self.budget:
                break

            for i in range(pop_size):
                indices = list(range(pop_size))
                indices.remove(i)
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + 0.8 * (b - c), lb, ub)
                trial = np.where(np.random.rand(len(lb)) < 0.9, mutant, pop[i])
                trial_fitness = func(trial)

                if len(fitness) >= self.budget:
                    break

                # Stochastic acceptance for exploration
                if trial_fitness < fitness[i] or np.random.rand() < 0.05:
                    pop[i], fitness[i] = trial, trial_fitness

        return pop, fitness

    def local_search(self, func, solution, bounds):
        best_solution = solution
        best_fitness = func(solution)
        step_size = (bounds.ub - bounds.lb) / 100.0

        for _ in range(self.local_refinement_steps):
            neighbors = [solution + step_size * np.random.randn(self.dim) for _ in range(5)]
            for neighbor in neighbors:
                if bounds.lb <= neighbor.all() <= bounds.ub:
                    neighbor_fitness = func(neighbor)
                    if neighbor_fitness < best_fitness:
                        best_solution, best_fitness = neighbor, neighbor_fitness
                        step_size *= 0.5  # Adaptive step size reduction for more precise refinement
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