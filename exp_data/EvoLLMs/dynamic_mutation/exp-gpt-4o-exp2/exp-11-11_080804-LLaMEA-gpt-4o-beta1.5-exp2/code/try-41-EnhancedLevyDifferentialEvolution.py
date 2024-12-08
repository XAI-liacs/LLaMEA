import numpy as np

class EnhancedLevyDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.f = 0.5
        self.cr = 0.9
        self.alpha = 1.5

    def levy_flight(self, size):
        u = np.random.normal(0, 1, size) * (np.sqrt(np.abs(np.random.normal(0, 1, size))) ** (-1 / self.alpha))
        return u

    def adaptive_mutation_scaling(self, evaluations):
        return self.f * (1 - (evaluations / self.budget))

    def dynamic_crossover_probability(self, evaluations):
        return self.cr * (0.7 + 0.3 * (self.budget - evaluations) / self.budget)

    def adaptive_noise_level(self, evaluations):
        return 0.1 * (1 - evaluations / self.budget)

    def adaptive_population_size(self, evaluations):
        return max(10, int(self.population_size * (1 - evaluations / self.budget)))

    def opposition_based_learning(self, individual):
        return self.lower_bound + self.upper_bound - individual

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                adaptive_f = self.adaptive_mutation_scaling(evaluations)
                noise_level = self.adaptive_noise_level(evaluations)
                noise = np.random.normal(0, noise_level, self.dim)
                mutant = np.clip(x0 + adaptive_f * (x1 - x2 + noise), self.lower_bound, self.upper_bound)
                
                crossover_prob = self.dynamic_crossover_probability(evaluations)
                crossover = np.random.rand(self.dim) < crossover_prob
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])
                
                step_size = (self.budget - evaluations) / self.budget
                levy_step = self.levy_flight(self.dim)
                trial += step_size * levy_step * np.exp(-evaluations / self.budget)
                trial = np.clip(trial, self.lower_bound, self.upper_bound)
                
                opp_trial = self.opposition_based_learning(trial)
                opp_trial_fitness = func(opp_trial)
                
                trial_fitness = func(trial)
                evaluations += 1
                if opp_trial_fitness < trial_fitness:
                    trial = opp_trial
                    trial_fitness = opp_trial_fitness
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best = trial

                if evaluations >= self.budget:
                    break

            self.population_size = self.adaptive_population_size(evaluations)
            population = np.resize(population, (self.population_size, self.dim))
            fitness = np.resize(fitness, self.population_size)

        return best