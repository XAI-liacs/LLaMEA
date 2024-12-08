import numpy as np

class EnhancedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 30  # Slightly larger initial population
        self.min_population_size = 10
        self.de_mutation_factor = 0.75  # Adjusted mutation factor for better exploration
        self.cr = 0.85  # Adjusted crossover rate for balance
        self.initial_temperature = 120.0  # Higher starting temperature for diversity
        self.temperature_decay = 0.95  # Slower decay for prolonged exploration

    def __call__(self, func):
        np.random.seed(42)
        population_size = self.initial_population_size
        # Hybrid initialization for enhanced search space coverage
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals_used = population_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        def de_mutation_and_crossover(target_idx, temperature):
            indices = list(range(population_size))
            indices.remove(target_idx)
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.de_mutation_factor * (b - c), self.lower_bound, self.upper_bound)
            cross_points = np.random.rand(self.dim) < self.cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[target_idx])
            # Gradient-based local refinement for trial solution
            gradient_step = 0.01 * np.sign(trial - population[target_idx])
            trial = np.clip(trial + gradient_step, self.lower_bound, self.upper_bound)
            return trial

        temperature = self.initial_temperature
        while evals_used < self.budget:
            for i in range(population_size):
                trial = de_mutation_and_crossover(i, temperature)
                trial_fitness = func(trial)
                evals_used += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evals_used >= self.budget:
                    break

            # Dynamic adjustment of population size based on convergence
            population_size = max(self.min_population_size, int(self.initial_population_size * (1 - evals_used / self.budget)))
            population = population[:population_size]
            fitness = fitness[:population_size]

            temperature *= self.temperature_decay

        return best_solution, best_fitness