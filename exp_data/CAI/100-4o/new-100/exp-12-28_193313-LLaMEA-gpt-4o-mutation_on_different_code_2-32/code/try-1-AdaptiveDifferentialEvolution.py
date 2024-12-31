import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.cr = cr
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        f_best = np.min(fitness)
        x_best = population[np.argmin(fitness)]
        
        mutation_factor = 0.5
        mutation_factor_min = 0.1
        mutation_factor_max = 0.9
        success_rate = 0.1  # Initial success rate
        decay_factor = 0.99  # Added decay factor

        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]

                mutant_vector = x1 + mutation_factor * (x2 - x3)
                mutant_vector = np.clip(mutant_vector, lb, ub)

                trial_vector = np.copy(population[i])
                crossover = np.random.rand(self.dim) < self.cr
                trial_vector[crossover] = mutant_vector[crossover]

                f_trial = func(trial_vector)
                eval_count += 1

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial_vector

                    if f_trial < f_best:
                        f_best = f_trial
                        x_best = trial_vector

            success_rate = np.mean(fitness < f_best)
            mutation_factor = (mutation_factor_min + (mutation_factor_max - mutation_factor_min) * success_rate) * decay_factor

        self.f_opt = f_best
        self.x_opt = x_best
        return self.f_opt, self.x_opt