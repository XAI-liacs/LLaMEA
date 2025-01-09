import numpy as np

class DE_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.cr = 0.9
        self.f = 0.8
        self.Tmax = 1.0
        self.Tmin = 1e-5
        self.alpha = 0.9

    def __call__(self, func):
        def mutate(x, pop, f):
            a, b, c = np.random.choice(pop, 3, replace=False)
            mutant = np.clip(a + f * (b - c), -5.0, 5.0)
            return mutant

        def annealing_acceptance(x, new_x, temp):
            if func(new_x) < func(x) or np.random.rand() < np.exp((func(x) - func(new_x)) / temp):
                return new_x
            else:
                return x

        def annealing_schedule(t):
            return self.Tmax * (self.alpha**t)

        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        best_solution = pop[np.argmin([func(x) for x in pop])]
        t = 0
        while t < self.budget:
            new_pop = []
            for x in pop:
                mutant = mutate(x, pop, self.f)
                new_x = annealing_acceptance(x, mutant, annealing_schedule(t))
                new_pop.append(new_x)
                t += 1
                if t >= self.budget:
                    break
            pop = np.array(new_pop)
            best_solution = pop[np.argmin([func(x) for x in pop])]
        return best_solution