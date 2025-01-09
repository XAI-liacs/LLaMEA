import numpy as np

class HybridDE_LocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)

    def __call__(self, func):
        eval_count = 0
        while eval_count < self.budget:
            # Evaluate current population
            for i in range(self.population_size):
                if self.fitness[i] == np.inf:  # Only evaluate unevaluated individuals
                    self.fitness[i] = func(self.population[i])
                    eval_count += 1
                    if eval_count >= self.budget:
                        return self.get_best_solution()

            # Differential Evolution Mutation and Crossover
            for i in range(self.population_size):
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = np.clip(self.population[a] + self.mutation_factor * (self.population[b] - self.population[c]), self.lower_bound, self.upper_bound)
                trial = np.copy(self.population[i])
                crossover_points = np.random.rand(self.dim) < self.crossover_rate
                trial[crossover_points] = mutant[crossover_points]
                trial_fitness = func(trial)
                eval_count += 1

                # Selection
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                if eval_count >= self.budget:
                    return self.get_best_solution()

            # Local Search on Best Individual
            best_idx = np.argmin(self.fitness)
            best_individual = self.population[best_idx]
            local_best = self.local_search(func, best_individual)
            local_best_fitness = func(local_best)
            eval_count += 1
            if local_best_fitness < self.fitness[best_idx]:
                self.population[best_idx] = local_best
                self.fitness[best_idx] = local_best_fitness

        return self.get_best_solution()

    def local_search(self, func, individual):
        step_size = 0.1
        for _ in range(10):
            neighbor = individual + np.random.uniform(-step_size, step_size, self.dim)
            neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
            if func(neighbor) < func(individual):
                individual = neighbor
        return individual

    def get_best_solution(self):
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]