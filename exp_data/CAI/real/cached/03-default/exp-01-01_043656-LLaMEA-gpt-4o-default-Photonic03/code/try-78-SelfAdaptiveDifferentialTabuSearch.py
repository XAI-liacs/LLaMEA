import numpy as np

class SelfAdaptiveDifferentialTabuSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.tabu_size = int(self.population_size / 5)
        self.mutation_factor = 0.5
        self.recombination_rate = 0.7

    def mutate(self, population, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = population[a] + self.mutation_factor * (population[b] - population[c])
        return mutant

    def crossover(self, target, mutant):
        crossover = np.random.rand(self.dim) < self.recombination_rate
        trial = np.where(crossover, mutant, target)
        return trial

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        tabu_list = []
        best_solution = None
        best_value = float('inf')
        
        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(pop, i)
                mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])
                trial = self.crossover(pop[i], mutant)
                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])
                
                trial_value = func(trial)
                eval_count += 1
                
                if len(tabu_list) >= self.tabu_size:
                    tabu_list.pop(0)
                
                if trial_value < best_value and not any(np.allclose(trial, tabu) for tabu in tabu_list):
                    best_solution = trial
                    best_value = trial_value
                    tabu_list.append(trial)
                
                if trial_value < func(pop[i]):
                    pop[i] = trial

                if eval_count >= self.budget:
                    break

        return best_solution