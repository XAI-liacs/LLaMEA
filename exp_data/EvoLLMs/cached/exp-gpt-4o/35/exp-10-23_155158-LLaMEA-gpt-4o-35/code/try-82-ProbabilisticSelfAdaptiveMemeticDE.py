import numpy as np

class ProbabilisticSelfAdaptiveMemeticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 * dim  # Increased initial population size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_factor = 0.9  # Self-adaptive mutation factor
        self.crossover_prob = 0.9  # Self-adaptive crossover probability
        self.success_rate_history = []
        self.memory_size = 10  # Adjusted memory size
        self.memory = np.zeros((self.memory_size, 2))

    def adapt_parameters(self):
        if len(self.success_rate_history) >= self.memory_size:
            recent_success = np.mean(self.success_rate_history[-self.memory_size:])
            base_factor = 0.6 + 0.4 * recent_success  # Adjusted base factor
            self.mutation_factor = np.clip(base_factor + 0.15 * np.random.randn(), 0.4, 1.3)
            self.crossover_prob = np.clip(0.8 + 0.2 * recent_success + 0.07 * np.random.randn(), 0.5, 1.0)
            self.memory = np.roll(self.memory, -1, axis=0)
            self.memory[-1] = [self.mutation_factor, self.crossover_prob]

    def intensification(self, best_individual):
        step_size = 0.015 + 0.035 * np.random.randn() * np.random.choice([1, -1])
        perturbation = np.random.uniform(-step_size, step_size, self.dim)
        candidate = np.clip(best_individual + perturbation, self.lower_bound, self.upper_bound)
        return candidate

    def dynamic_population_control(self, generation):
        if generation % 3 == 0 and self.population_size > 14:  # Dynamic population control
            self.population_size = max(14, int(0.7 * self.population_size))

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size
        generation = 0

        while budget_used < self.budget:
            new_population = []
            new_fitness = []
            self.dynamic_population_control(generation)

            for i in range(self.population_size):
                idxs = np.delete(np.arange(self.population_size), i)
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                if np.random.rand() < 0.35:  # Probabilistic mutation
                    mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                else:
                    d = population[np.random.choice(idxs, 1, replace=False)][0]
                    mutant_vector = np.clip(a + self.mutation_factor * (b - c + d), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(cross_points, mutant_vector, population[i])
                trial_fitness = func(trial_vector)
                budget_used += 1

                if trial_fitness < fitness[i]:
                    new_population.append(trial_vector)
                    new_fitness.append(trial_fitness)
                    self.success_rate_history.append(1)
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])
                    self.success_rate_history.append(0)

                if budget_used >= self.budget:
                    break

            population = np.array(new_population)
            fitness = np.array(new_fitness)
            self.adapt_parameters()

            best_idx = np.argmin(fitness)
            if budget_used < self.budget and np.random.rand() < 0.6:  # Targeted intensification
                intense_candidate = self.intensification(population[best_idx])
                intense_fitness = func(intense_candidate)
                budget_used += 1

                if intense_fitness < fitness[best_idx]:
                    population[best_idx] = intense_candidate
                    fitness[best_idx] = intense_fitness

            generation += 1

            if budget_used >= self.budget:
                break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]