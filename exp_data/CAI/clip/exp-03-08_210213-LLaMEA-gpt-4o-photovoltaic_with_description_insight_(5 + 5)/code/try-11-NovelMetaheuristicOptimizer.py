import numpy as np

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, dim * 10)
        self.crossover_rate = 0.9
        self.mutation_factor = 0.8
        self.local_refinement_steps = 5
        self.layers_increment = max(int(dim / 10), 1)
        self.robustness_tolerance = 1e-3

    def differential_evolution(self, population, func, bounds):
        new_population = np.copy(population)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.mutation_factor * (b - c), bounds[0], bounds[1])
            cross_points = np.random.rand(self.dim) < self.crossover_rate
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            if func(trial) < func(population[i]):
                new_population[i] = trial
        return new_population

    def local_search(self, solution, func, bounds):
        best_solution = np.copy(solution)
        best_score = func(best_solution)
        learning_rate = 0.01 * (self.budget - self.local_refinement_steps) / self.budget  # Adaptive learning rate
        for _ in range(self.local_refinement_steps):
            grad = (func(best_solution + self.robustness_tolerance) - best_score) / self.robustness_tolerance
            candidate = np.clip(best_solution - learning_rate * grad, bounds[0], bounds[1])
            candidate_score = func(candidate)
            if candidate_score < best_score:
                best_solution, best_score = candidate, candidate_score
        return best_solution

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        evaluations = 0
        
        while evaluations < self.budget:
            population = self.differential_evolution(population, func, bounds)
            for i in range(self.population_size):
                population[i] = self.local_search(population[i], func, bounds)
                evaluations += 1
                if evaluations >= self.budget:
                    break
            current_dim = min(self.dim, (evaluations // self.layers_increment) + 1)
            if current_dim != self.dim:
                extended_population = np.random.uniform(bounds[0], bounds[1], (self.population_size, current_dim))
                extended_population[:, :self.dim] = population[:, :self.dim]
                population = extended_population
                self.dim = current_dim

        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]