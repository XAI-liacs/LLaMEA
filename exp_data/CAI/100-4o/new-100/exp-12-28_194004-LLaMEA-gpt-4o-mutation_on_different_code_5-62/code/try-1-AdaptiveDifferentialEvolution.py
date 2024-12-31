import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.history = []

    def _initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
    
    def _mutation(self, pop, F=0.5):
        idxs = np.random.choice(self.pop_size, 3, replace=False)
        return pop[idxs[0]] + F * (pop[idxs[1]] - pop[idxs[2]])

    def _crossover(self, target, mutant, Cr=0.7):
        j_rand = np.random.randint(self.dim)
        trial = np.where(np.random.rand(self.dim) < Cr, mutant, target)
        trial[j_rand] = mutant[j_rand]  # Ensure at least one parameter from mutant
        return trial

    def _adaptive_parameters(self):
        if len(self.history) == 0:
            return 0.5, 0.7  # Initial F and Cr
        else:
            success_rate = sum(f < self.f_opt for f in self.history[-10:]) / 10.0
            F = np.clip(0.5 * (1 + success_rate), 0.4, 0.9)
            Cr = np.clip(0.7 * (1 + success_rate), 0.5, 0.9)
            self.pop_size = int(np.clip(self.pop_size * (1 + success_rate), 20, 100))  # Dynamic population size
            return F, Cr

    def __call__(self, func):
        bounds = func.bounds
        pop = self._initialize_population(bounds)
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.pop_size

        while eval_count < self.budget:
            F, Cr = self._adaptive_parameters()
            new_population = np.empty_like(pop)

            for i in range(self.pop_size):
                mutant = self._mutation(pop, F=F)
                trial = self._crossover(pop[i], mutant, Cr=Cr)

                trial = np.clip(trial, bounds.lb, bounds.ub)
                f_trial = func(trial)
                eval_count += 1

                if f_trial < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = f_trial
                    self.history.append(f_trial)
                else:
                    new_population[i] = pop[i]

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if eval_count >= self.budget:
                    break

            pop = new_population

        return self.f_opt, self.x_opt