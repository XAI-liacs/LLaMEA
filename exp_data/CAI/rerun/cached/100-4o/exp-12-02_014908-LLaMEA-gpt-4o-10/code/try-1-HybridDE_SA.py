import numpy as np

class HybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf
        self.evaluations = 0
        self.F = 0.5 # Differential weight
        self.CR = 0.9 # Crossover probability

    def differential_evolution(self, target_idx):
        idxs = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant_vector = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
        trial_vector = np.array([
            mutant_vector[i] if np.random.rand() < self.CR else self.population[target_idx][i]
            for i in range(self.dim)
        ])
        return trial_vector

    def simulated_annealing(self, candidate, candidate_fitness):
        t_initial = 1.0
        t_final = 0.01
        alpha = 0.99
        t = t_initial
        current_solution = candidate
        current_fitness = candidate_fitness

        while t > t_final:
            neighbor = current_solution + np.random.uniform(-0.1, 0.1, self.dim)
            neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
            neighbor_fitness = self.evaluate(neighbor)

            if neighbor_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - neighbor_fitness) / t):
                current_solution = neighbor
                current_fitness = neighbor_fitness

            t *= alpha

        return current_solution, current_fitness

    def evaluate(self, solution):
        if self.evaluations >= self.budget:
            return np.inf
        self.evaluations += 1
        return solution.func(solution)

    def __call__(self, func):
        self.best_solution = None
        self.best_fitness = np.inf
        self.evaluations = 0

        for i in range(self.population_size):
            self.fitness[i] = self.evaluate(self.population[i])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.population[i].copy()

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                trial_vector = self.differential_evolution(i)
                trial_fitness = self.evaluate(trial_vector)

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness

                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial_vector

                # Apply simulated annealing for further local refinement
                refined_solution, refined_fitness = self.simulated_annealing(self.population[i], self.fitness[i])
                self.population[i] = refined_solution
                self.fitness[i] = refined_fitness

                if refined_fitness < self.best_fitness:
                    self.best_fitness = refined_fitness
                    self.best_solution = refined_solution

                if self.evaluations >= self.budget:
                    break

        return self.best_solution, self.best_fitness