import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_iter = budget // self.population_size

    def __call__(self, func):
        def fitness(x):
            return func(x)

        def initialize_population():
            return np.random.uniform(-5, 5, (self.population_size, self.dim))

        def pso(particles):
            # Particle Swarm Optimization
            c1 = 2.0
            c2 = 2.0
            w = 0.7
            pbest = particles.copy()
            pbest_fit = np.array([fitness(p) for p in pbest])
            gbest_idx = np.argmin(pbest_fit)
            gbest = pbest[gbest_idx].copy()

            for _ in range(self.max_iter):
                r1 = np.random.rand(self.population_size, self.dim)
                r2 = np.random.rand(self.population_size, self.dim)
                v = w * v + c1 * r1 * (pbest - particles) + c2 * r2 * (gbest - particles)
                particles = particles + v
                particles = np.clip(particles, -5, 5)
                current_fit = np.array([fitness(p) for p in particles])
                update_indices = current_fit < pbest_fit
                pbest[update_indices] = particles[update_indices]
                pbest_fit[update_indices] = current_fit[update_indices]
                gbest_idx = np.argmin(pbest_fit)
                gbest = pbest[gbest_idx]

            return gbest

        def de(population):
            # Differential Evolution
            cr = 0.9
            f = 0.8
            scale_factor = 0.5
            bounds = [(-5, 5)] * self.dim
            for _ in range(self.max_iter):
                for i in range(self.population_size):
                    target = population[i]
                    indices = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = population[np.random.choice(indices, 3, replace=False)]
                    mutant = np.clip(a + f * (b - c), -5, 5)
                    crossover = np.random.rand(self.dim) < cr
                    trial = np.where(crossover, mutant, target)
                    if fitness(trial) < fitness(target):
                        population[i] = trial

            return population[np.argmin([fitness(p) for p in population])]

        best_solution = np.zeros(self.dim)
        population = initialize_population()

        for _ in range(self.max_iter):
            new_solutions = np.array([pso(p) for p in population])
            population = np.array([de(p) for p in new_solutions])

        best_solution = population[np.argmin([fitness(p) for p in population])]
        return best_solution