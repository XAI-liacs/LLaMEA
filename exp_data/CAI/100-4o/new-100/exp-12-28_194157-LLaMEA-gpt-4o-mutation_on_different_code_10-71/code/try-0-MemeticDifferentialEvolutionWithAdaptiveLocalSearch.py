import numpy as np

class MemeticDifferentialEvolutionWithAdaptiveLocalSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 50
        self.crossover_rate = 0.9
        self.mutation_factor = 0.8
        self.local_search_budget = 5

    def differential_evolution(self, population, func):
        trial_population = np.copy(population)
        for i in range(self.population_size):
            # Mutation
            a, b, c = np.random.choice(self.population_size, 3, replace=False)
            mutant = population[a] + self.mutation_factor * (population[b] - population[c])
            mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
            
            # Crossover
            cross_points = np.random.rand(self.dim) < self.crossover_rate
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            # Selection
            if func(trial) < func(population[i]):
                trial_population[i] = trial
                if func(trial) < self.f_opt:
                    self.f_opt = func(trial)
                    self.x_opt = trial
        return trial_population

    def adaptive_local_search(self, x, func):
        step_size = 0.1
        for _ in range(self.local_search_budget):
            perturbation = step_size * np.random.uniform(-1, 1, self.dim)
            candidate = np.clip(x + perturbation, func.bounds.lb, func.bounds.ub)
            if func(candidate) < func(x):
                x = candidate
                if func(x) < self.f_opt:
                    self.f_opt = func(x)
                    self.x_opt = x
            step_size *= 0.9  # Gradually reduce step size for convergence
        return x

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        evaluation_count = 0

        # Evaluate initial population
        for i in range(self.population_size):
            f = func(population[i])
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = population[i]
            evaluation_count += 1
            if evaluation_count >= self.budget:
                return self.f_opt, self.x_opt

        # Main loop
        while evaluation_count < self.budget:
            # Differential Evolution phase
            population = self.differential_evolution(population, func)
            evaluation_count += self.population_size
            if evaluation_count >= self.budget:
                break
            
            # Adaptive Local Search phase
            for i in range(self.population_size):
                population[i] = self.adaptive_local_search(population[i], func)
                evaluation_count += self.local_search_budget
                if evaluation_count >= self.budget:
                    break

        return self.f_opt, self.x_opt