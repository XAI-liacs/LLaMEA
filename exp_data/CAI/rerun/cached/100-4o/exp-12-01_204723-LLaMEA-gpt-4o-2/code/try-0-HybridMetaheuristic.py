import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.num_evaluations = 0

    def differential_evolution(self, func, population):
        for i in range(self.population_size):
            if self.num_evaluations >= self.budget:
                break
            indices = np.random.choice(self.population_size, 3, replace=False)
            x1, x2, x3 = population[indices]
            mutant_vector = np.clip(x1 + self.mutation_factor * (x2 - x3), self.lower_bound, self.upper_bound)
            trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, population[i])
            if self.num_evaluations < self.budget:
                trial_value = func(trial_vector)
                self.num_evaluations += 1
                if trial_value < population[i][1]:
                    population[i] = (trial_vector, trial_value)

    def local_search(self, func, point):
        step_size = 0.1
        for _ in range(10):  # small number of iterations for local refinement
            if self.num_evaluations >= self.budget:
                break
            candidate = point + step_size * np.random.randn(self.dim)
            candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
            if self.num_evaluations < self.budget:
                candidate_value = func(candidate)
                self.num_evaluations += 1
                if candidate_value < point[1]:
                    point = (candidate, candidate_value)
        return point

    def __call__(self, func):
        # Initialize the population
        population = [(np.random.uniform(self.lower_bound, self.upper_bound, self.dim), None) for _ in range(self.population_size)]
        for i in range(self.population_size):
            if self.num_evaluations < self.budget:
                population[i] = (population[i][0], func(population[i][0]))
                self.num_evaluations += 1

        # Optimization loop with DE and local search
        while self.num_evaluations < self.budget:
            self.differential_evolution(func, population)
            for i in range(self.population_size):
                if self.num_evaluations >= self.budget:
                    break
                population[i] = self.local_search(func, population[i])

        # Return the best solution found
        best_solution = min(population, key=lambda x: x[1])
        return best_solution[0]