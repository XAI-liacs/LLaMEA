import numpy as np

class Improved_EA_ADMSE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.cr = 0.5
        self.f = 0.5
        self.strategy_probs = np.full(6, 1 / 6)
        self.mut_probs = np.full(3, 0.5)  # Initialize mutation probabilities for different scaling factors
        self.cross_prob = 0.9

    def __call__(self, func):
        def mutate(x, pop, strategy):
            idxs = np.random.choice(len(pop), 3, replace=False)
            a, b, c = pop[idxs]
            return x + self.f * (b - c) * self.mut_probs[strategy]

        def crossover(x, trial):
            jrand = np.random.randint(self.dim)
            return np.where(np.arange(self.dim) == jrand, trial, x)

        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(p) for p in pop])

        for _ in range(self.budget):
            new_pop = np.empty_like(pop)
            for i in range(self.pop_size):
                strategy = np.random.choice(6, p=self.strategy_probs)
                trial = mutate(pop[i], pop, strategy)
                trial = np.clip(trial, -5.0, 5.0)
                trial = crossover(pop[i], trial)
                trial_fit = func(trial)
                if trial_fit < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = trial_fit
                    if strategy != 1:
                        self.strategy_probs[strategy] += 0.1
                        self.strategy_probs[self.strategy_probs < 1e-6] = 1e-6
                        self.strategy_probs /= np.sum(self.strategy_probs)
                    if trial_fit < fitness[i]:  
                        self.mut_probs[strategy % 3] = min(1.0, self.mut_probs[strategy % 3] * 1.2)
                        self.cross_prob = max(0.1, self.cross_prob * 0.8)
                else:
                    new_pop[i] = pop[i]
            pop = new_pop

        best_idx = np.argmin(fitness)
        return pop[best_idx]