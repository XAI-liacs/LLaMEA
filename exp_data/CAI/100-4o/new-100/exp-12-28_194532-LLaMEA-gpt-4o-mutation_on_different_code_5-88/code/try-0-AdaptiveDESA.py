import numpy as np

class AdaptiveDESA:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.temperature = 100.0

    def _initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def _mutate(self, x, population, lb, ub):
        idxs = np.random.choice(len(population), 3, replace=False)
        a, b, c = population[idxs]
        mutant = np.clip(a + 0.5 * (b - c), lb, ub)
        return mutant
    
    def _crossover(self, target, mutant):
        crossover_prob = 0.8
        mask = np.random.rand(self.dim) < crossover_prob
        trial = np.where(mask, mutant, target)
        return trial

    def _acceptance(self, f_new, f_old):
        if f_new < f_old:
            return True
        else:
            return np.random.rand() < np.exp((f_old - f_new) / self.temperature)
    
    def _temperature_decay(self, factor=0.99):
        self.temperature *= factor

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self._initialize_population(lb, ub)
        fitness = np.array([func(ind) for ind in population])
        
        eval_count = len(population)

        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                target = population[i]
                mutant = self._mutate(target, population, lb, ub)
                trial = self._crossover(target, mutant)
                f_new = func(trial)
                eval_count += 1

                if self._acceptance(f_new, fitness[i]):
                    population[i] = trial
                    fitness[i] = f_new
                    if f_new < self.f_opt:
                        self.f_opt = f_new
                        self.x_opt = trial
            
            self._temperature_decay()

        return self.f_opt, self.x_opt