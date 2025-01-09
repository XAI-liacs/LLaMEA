import numpy as np

class EnhancedHybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20 + int(3 * np.log(dim))  # Empirical choice for population size
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.adaptive_factor = 0.1  # Factor for hybridization of strategies
        self.initial_F = 0.5  # Initial mutation factor
        self.initial_CR = 0.9  # Initial crossover rate

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            new_population = []
            F = self.initial_F * (1 - evaluations / self.budget)  # Dynamic mutation factor
            CR = self.initial_CR * (1 - evaluations / self.budget)  # Dynamic crossover rate
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                
                # Mutation
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)  # Ensure bounds
                
                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, self.population[i])
                
                # Fitness evaluation
                f_trial = func(trial)
                evaluations += 1
                
                # Selection
                if f_trial < func(self.population[i]):
                    new_population.append(trial)
                    if f_trial < self.best_fitness:
                        self.best_fitness = f_trial
                        self.best_solution = trial
                else:
                    new_population.append(self.population[i])

            # Adaptive strategy with swarm behavior
            if np.random.rand() < self.adaptive_factor:
                swarm_center = np.mean(new_population, axis=0)
                for j in range(self.population_size):
                    perturbation = np.random.normal(0, 1, self.dim)
                    candidate = swarm_center + perturbation * (self.upper_bound - self.lower_bound) * 0.1
                    candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                    f_candidate = func(candidate)
                    evaluations += 1
                    if f_candidate < self.best_fitness:
                        self.best_fitness = f_candidate
                        self.best_solution = candidate
                    if f_candidate < func(new_population[j]):
                        new_population[j] = candidate

            self.population = np.array(new_population)

        return self.best_solution, self.best_fitness