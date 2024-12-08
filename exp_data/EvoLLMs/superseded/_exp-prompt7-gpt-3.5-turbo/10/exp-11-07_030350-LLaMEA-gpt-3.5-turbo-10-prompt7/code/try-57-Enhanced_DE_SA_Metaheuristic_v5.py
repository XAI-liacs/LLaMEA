import numpy as np

class Enhanced_DE_SA_Metaheuristic_v5:
    def __init__(self, budget, dim):
        self.budget, self.dim, self.pop_size, self.CR, self.alpha = budget, dim, 10, 0.5, 0.9
        self.F, self.T, self.population = 0.5, 1.0, np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

    def __call__(self, func):
        best_solution, best_fitness, F_init = np.random.uniform(-5.0, 5.0, self.dim), func(np.zeros(self.dim)), self.F

        for _ in range(self.budget):
            new_population, adapt_F = [], np.clip(F_init + 0.1 * np.random.randn(), 0.1, 0.9)
            for target in self.population:
                mutant_indices = np.random.choice(self.pop_size, 2, replace=False)
                mutant = self.population[mutant_indices]
                trial = target + adapt_F * (mutant[0] - mutant[1])
                mask = np.random.rand(self.dim) < self.CR
                trial[mask] = np.where(mask, trial, target)

                new_fitness = func(trial)
                if new_fitness < best_fitness:
                    best_solution, best_fitness = trial, new_fitness

                target = np.where(new_fitness < func(target) | (np.exp((func(target) - new_fitness) / self.T) > np.random.rand()), trial, target)
                new_population.append(target)

            self.population, self.T = np.array(new_population), self.T * max(self.alpha, 0.1)

        return best_solution