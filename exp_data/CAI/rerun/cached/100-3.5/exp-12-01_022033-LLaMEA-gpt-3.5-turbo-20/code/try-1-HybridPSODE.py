import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim, swarm_size=30, cr=0.9, f=0.8, w=0.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.cr = cr
        self.f = f
        self.w = w

    def __call__(self, func):
        def initialize_swarm():
            return np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))

        def clipToBounds(x):
            return np.clip(x, -5.0, 5.0)

        def DE(x, a, b, c):
            mutant = clipToBounds(a + self.f * (b - c))
            return mutant

        swarm = initialize_swarm()
        fitness = [func(x) for x in swarm]
        best_idx = np.argmin(fitness)
        best_solution = swarm[best_idx]

        for _ in range(self.budget - self.swarm_size):
            for i in range(self.swarm_size):
                a, b, c = np.random.choice(swarm, 3, replace=False)
                mutant = DE(swarm[i], a, b, c)

                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, swarm[i])
                trial_fitness = func(trial)

                if trial_fitness < fitness[i]:
                    swarm[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < func(best_solution):
                        best_solution = trial

        return best_solution