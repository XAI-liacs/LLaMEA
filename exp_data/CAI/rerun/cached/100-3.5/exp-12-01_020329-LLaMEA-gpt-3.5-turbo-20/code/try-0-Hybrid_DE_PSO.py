import numpy as np

class Hybrid_DE_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.cr = 0.7
        self.f = 0.5
        self.w = 0.7
        self.c1 = 1.5
        self.c2 = 1.5

    def __call__(self, func):
        def de_rand_1_bin(x, pop, f):
            r1, r2, r3 = np.random.choice(pop, 3, replace=False)
            return np.clip(r1 + f * (r2 - r3), -5.0, 5.0)

        def pso_update(x, pbest, gbest):
            v = self.w * x['velocity'] + self.c1 * np.random.rand(self.dim) * (pbest - x['position']) + \
                self.c2 * np.random.rand(self.dim) * (gbest - x['position'])
            x['velocity'] = np.clip(v, -1.0, 1.0)
            x['position'] = np.clip(x['position'] + x['velocity'], -5.0, 5.0)
            return x

        population = [{'position': np.random.uniform(-5.0, 5.0, self.dim), 'velocity': np.zeros(self.dim)} 
                      for _ in range(self.pop_size)]
        pbest = np.array([p['position'] for p in population])
        gbest = pbest[np.argmin([func(p) for p in pbest])]

        for _ in range(self.budget - self.pop_size):
            for i, x in enumerate(population):
                trial = de_rand_1_bin(x['position'], [p['position'] for p in population if p != x], self.f)
                trial = pso_update({'position': trial, 'velocity': x['velocity']}, pbest[i], gbest)
                if func(trial['position']) < func(x['position']):
                    x['position'] = trial['position']
                    pbest[i] = trial['position']
                    if func(trial['position']) < func(gbest):
                        gbest = trial['position']

        return gbest