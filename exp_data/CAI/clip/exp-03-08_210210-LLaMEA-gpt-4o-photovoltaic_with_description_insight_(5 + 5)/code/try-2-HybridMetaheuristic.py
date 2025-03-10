import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20  # Initial population size
        self.f = 0.5  # DE scaling factor
        self.cr = 0.9  # Crossover probability
        self.num_evals = 0  # Track the number of function evaluations

    def differential_evolution(self, pop, bounds, func):
        for i in range(len(pop)):
            idxs = [idx for idx in range(len(pop)) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + self.f * (b - c)
            mutant = np.clip(mutant, bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < self.cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            if self.num_evals < self.budget:
                trial_fitness = func(trial)
                self.num_evals += 1
                if trial_fitness > func(pop[i]):
                    pop[i] = trial
        return pop

    def local_search(self, solution, bounds, func, complexity_increment):
        perturbed_solution = solution.copy()
        complexity = complexity_increment
        while complexity < self.dim:
            layer_indices = np.random.choice(self.dim, complexity, replace=False)
            perturbation = np.random.uniform(-0.05, 0.05, size=complexity)
            perturbed_solution[layer_indices] += perturbation
            perturbed_solution = np.clip(perturbed_solution, bounds.lb, bounds.ub)
            if self.num_evals < self.budget:
                perturbed_fitness = func(perturbed_solution)
                self.num_evals += 1
                if perturbed_fitness > func(solution):
                    solution = perturbed_solution.copy()
            complexity += complexity_increment
        return solution

    def __call__(self, func):
        bounds = func.bounds
        pop = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        pop_fitness = np.array([func(indiv) for indiv in pop])
        self.num_evals += self.population_size

        while self.num_evals < self.budget:
            pop = self.differential_evolution(pop, bounds, func)
            best_idx = np.argmax(pop_fitness)
            best_solution = pop[best_idx]
            best_solution = self.local_search(best_solution, bounds, func, complexity_increment=5)
            if self.num_evals >= self.budget:
                break

        best_idx = np.argmax(pop_fitness)
        return pop[best_idx]