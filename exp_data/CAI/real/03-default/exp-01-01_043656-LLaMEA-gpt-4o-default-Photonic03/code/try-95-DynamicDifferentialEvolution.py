import numpy as np

class DynamicDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.F_min = 0.5
        self.F_max = 0.9
        self.CR_min = 0.1
        self.CR_max = 0.9

    def adapt_parameters(self, diversity):
        F = self.F_min + (self.F_max - self.F_min) * (1 - diversity)
        CR = self.CR_min + (self.CR_max - self.CR_min) * diversity
        return F, CR

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size

        while eval_count < self.budget:
            diversity = np.mean(np.std(pop, axis=0) / (bounds[:, 1] - bounds[:, 0]))
            F, CR = self.adapt_parameters(diversity)

            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])

                trial = np.copy(pop[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                trial_value = func(trial)
                eval_count += 1
                if trial_value < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_value

                if eval_count >= self.budget:
                    break

        best_index = np.argmin(fitness)
        return pop[best_index]