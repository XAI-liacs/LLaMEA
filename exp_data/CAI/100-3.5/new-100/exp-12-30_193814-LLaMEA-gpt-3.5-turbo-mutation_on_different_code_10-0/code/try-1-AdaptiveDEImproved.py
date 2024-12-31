import numpy as np

class AdaptiveDEImproved:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10
        F = 0.5
        CR = 0.9
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))

        for _ in range(self.budget):
            trial_population = []
            diversity = np.std(population, axis=0)
            F = 0.5 + 0.3 * np.tanh(np.mean(diversity) - np.min(diversity))

            for i in range(pop_size):
                a, b, c = np.random.choice(population, 3, replace=False)
                mutant_vector = np.clip(a + F * (b - c), bounds[0], bounds[1])
                crossover_points = np.random.rand(self.dim) < CR
                trial_vector = np.where(crossover_points, mutant_vector, population[i])

                trial_population.append(trial_vector)

            for trial_vec in trial_population:
                f = func(trial_vec)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial_vec

            population = trial_population

        return self.f_opt, self.x_opt