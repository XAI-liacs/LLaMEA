import numpy as np

class AdaptiveEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 30
        evals_per_iter = pop_size
        bounds = (-5.0, 5.0)
        cr = 0.9
        f = 0.8
        max_mut_step = 1.0
        min_mut_step = 0.01
        mut_step_size = 0.5 * (max_mut_step - min_mut_step)

        population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget // evals_per_iter):
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(population[a] + f * (population[b] - population[c]), bounds[0], bounds[1])
                mask = np.random.rand(self.dim) < cr
                new_ind = np.where(mask, mutant, population[i])
                new_fitness = func(new_ind)
                
                if new_fitness < fitness[i]:
                    population[i] = new_ind
                    fitness[i] = new_fitness
                    mut_step_size = max(min_mut_step, min(max_mut_step, mut_step_size * 0.85))
                else:
                    mut_step_size = max(min_mut_step, min(max_mut_step, mut_step_size * 1.15))
            
        best_idx = np.argmin(fitness)
        return population[best_idx]