import numpy as np

class DE_PSO_Hybrid:
    def __init__(self, budget, dim, pop_size=50, max_iter=1000, de_cr=0.5, de_f=0.8, pso_c1=2.0, pso_c2=2.0):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.de_cr = de_cr
        self.de_f = de_f
        self.pso_c1 = pso_c1
        self.pso_c2 = pso_c2

    def __call__(self, func):
        def de(x, f, cr, pop):
            mutant = x + f * (pop[np.random.choice(len(pop))] - pop[np.random.choice(len(pop))])
            trial = np.where(np.random.rand(self.dim) < cr, mutant, x)
            return trial

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        pbest_pos = population.copy()
        pbest_val = np.array([func(ind) for ind in population])
        gbest_idx = np.argmin(pbest_val)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_val = pbest_val[gbest_idx]

        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                new_pos = de(population[i], self.de_f, self.de_cr, population)
                new_val = func(new_pos)
                if new_val < pbest_val[i]:
                    pbest_val[i] = new_val
                    pbest_pos[i] = new_pos
                    if new_val < gbest_val:
                        gbest_val = new_val
                        gbest_pos = new_pos
                velocity = velocities[i] + self.pso_c1 * np.random.rand() * (pbest_pos[i] - population[i]) + self.pso_c2 * np.random.rand() * (gbest_pos - population[i])
                population[i] = population[i] + velocity
            if np.sum(np.abs(velocities)) < 1e-6:
                break

        return gbest_pos