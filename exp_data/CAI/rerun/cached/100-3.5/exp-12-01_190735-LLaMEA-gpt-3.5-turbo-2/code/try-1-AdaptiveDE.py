import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.NP = 10  # population size
        self.CR = 0.9  # crossover rate
        self.F = 0.5  # scaling factor
        self.strategy_probs = np.full((dim, 4), 0.25)  # initialize mutation strategy probabilities

    def __call__(self, func):
        def mutate(target, pop, strategy_probs):
            rand = np.random.rand(self.NP, 3)
            mutants = np.zeros((self.NP, self.dim))
            for i in range(self.NP):
                j = np.random.choice(range(4), p=strategy_probs[i])
                if j == 0:  # DE/rand/1
                    idxs = np.random.choice(range(self.NP), 3, replace=False)
                    mutants[i] = pop[idxs[0]] + self.F * (pop[idxs[1]] - pop[idxs[2]])
                elif j == 1:  # DE/current-to-best/2
                    idxs = np.argsort(func(pop))[0:2]
                    mutants[i] = target + self.F * (pop[idxs[0]] - pop[idxs[1]) + self.F * (pop[idxs[np.random.choice(range(2))] - pop[i]))
                elif j == 2:  # DE/current-to-rand/1
                    idxs = np.random.choice(range(self.NP), 2, replace=False)
                    mutants[i] = pop[i] + self.F * (pop[idxs[0]] - pop[i]) + self.F * (pop[idxs[1]] - pop[idxs[0]])
                else:  # DE/rand-to-best/2
                    idxs = np.argsort(func(pop))[0:2]
                    mutants[i] = pop[idxs[0]] + self.F * (pop[idxs[1]] - pop[idxs[0]) + self.F * (pop[np.random.choice(range(self.NP))] - pop[i]))
                for k in range(3):
                    if rand[i, k] < self.CR:
                        mutants[i, k] = target[k]
            return mutants

        bounds = (-5.0, 5.0)
        pop = np.random.uniform(bounds[0], bounds[1], (self.NP, self.dim))
        fitness = func(pop)
        best_idx = np.argmin(fitness)
        best = pop[best_idx].copy()

        for _ in range(self.budget):
            mutants = mutate(pop, pop, self.strategy_probs)
            mutants_fitness = func(mutants)
            for i in range(self.NP):
                if mutants_fitness[i] < fitness[i]:
                    pop[i] = mutants[i]
                    fitness[i] = mutants_fitness[i]
                    if mutants_fitness[i] < func(best):
                        best = mutants[i].copy()

            # Update strategy probabilities based on success of strategies
            successes = mutants_fitness < fitness
            for i in range(self.NP):
                j = np.argmax(np.random.multinomial(1, successes[i] * self.strategy_probs[i]))
                self.strategy_probs[i] = 0.9 * self.strategy_probs[i] + 0.1 * (j == np.arange(4))

        return best