import numpy as np

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def differential_evolution(self, func, pop_size, F=0.8, CR=0.9):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(pop_size, self.dim)
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        
        while self.evaluations < self.budget:
            for i in range(pop_size):
                indices = list(range(pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < CR
                CR = 0.5 + 0.4 * (1 - self.evaluations / self.budget)  # Dynamic CR update
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                f = func(trial)
                self.evaluations += 1
                if f < fitness[i]:
                    fitness[i] = f
                    population[i] = trial
                    if f < fitness[best_idx]:
                        best_idx = i
                        best_solution = trial
            
            if self.evaluations >= self.budget:
                break

        return best_solution, fitness[best_idx]

    def local_refinement(self, solution, func):
        step_size = 0.01 * (func.bounds.ub - func.bounds.lb)
        for _ in range(10):  # Local refinement iterations
            step_size *= 0.95  # Adaptive step-size reduction
            for i in range(self.dim):
                if self.evaluations >= self.budget:
                    break
                perturb = np.zeros(self.dim)
                perturb[i] = step_size[i]
                candidate = solution + perturb
                candidate_fitness = func(candidate)
                self.evaluations += 1
                if candidate_fitness < func(solution):
                    solution = candidate
                else:
                    candidate = solution - perturb
                    candidate_fitness = func(candidate)
                    self.evaluations += 1
                    if candidate_fitness < func(solution):
                        solution = candidate
        return solution

    def __call__(self, func):
        pop_size = 10 + self.dim * 2
        best_solution, best_fitness = self.differential_evolution(func, pop_size)
        best_solution = self.local_refinement(best_solution, func)
        return best_solution