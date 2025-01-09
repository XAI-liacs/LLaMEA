import numpy as np

class HEDHS:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.5, bw=0.01, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def harmony_search(self, func, population):
        for i in range(self.budget):
            new_pop = []
            for j in range(self.pop_size):
                if np.random.rand() < self.hmcr:
                    idx = np.random.choice(len(population))
                    new_member = population[idx] + np.random.uniform(-self.bw, self.bw, size=self.dim)
                else:
                    new_member = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)

                if np.random.rand() < self.par:
                    idx = np.random.choice(len(population))
                    new_member = population[idx] + self.par * (new_member - population[idx])

                f = func(new_member)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = new_member

                new_pop.append(new_member)

            population = np.array(new_pop)

        return self.f_opt, self.x_opt

    def differential_evolution(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        for i in range(self.budget):
            for j in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                trial = population[a] + self.par * (population[b] - population[c])

                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial

                if f < func(population[j]):
                    population[j] = trial

        return self.f_opt, self.x_opt

    def evolutionary_algorithm(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        for i in range(self.budget):
            new_pop = []
            for j in range(self.pop_size):
                idx = np.random.choice(len(population))
                new_member = population[idx] + np.random.uniform(-self.bw, self.bw, size=self.dim)

                f = func(new_member)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = new_member

                new_pop.append(new_member)

            population = np.array(new_pop)

        return self.f_opt, self.x_opt

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        # Perform hybrid optimization using HEDHS
        self.harmony_search(func, self.differential_evolution(func)[1])
        self.harmony_search(func, self.evolutionary_algorithm(func)[1])

        return self.f_opt, self.x_opt