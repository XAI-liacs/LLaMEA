import numpy as np

class ImprovedHybridPSODE:
    def __init__(self, budget, dim, swarm_size=30, mutation_factor=0.5, crossover_prob=0.7, w=0.5, c1=1.494, c2=1.494):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        def cost_function(x):
            return func(x)

        def create_particle():
            return np.random.uniform(-5.0, 5.0, self.dim)

        def pso():
            # PSO initialization
            swarm = np.array([create_particle() for _ in range(self.swarm_size)])
            pbest = swarm.copy()
            pbest_values = np.array([cost_function(p) for p in pbest])
            gbest = pbest[np.argmin(pbest_values)]
            gbest_value = np.min(pbest_values)
            velocity = np.zeros((self.swarm_size, self.dim))

            for _ in range(self.budget):
                r1, r2 = np.random.random(size=(2, self.swarm_size, self.dim))
                velocity = self.w * velocity + self.c1 * r1 * (pbest - swarm) + self.c2 * r2 * (gbest - swarm)
                swarm += velocity
                swarm = np.clip(swarm, -5.0, 5.0)
                values = np.array([cost_function(p) for p in swarm])

                # Update personal best
                update_idx = values < pbest_values
                pbest[update_idx] = swarm[update_idx]
                pbest_values[update_idx] = values[update_idx]

                # Update global best
                min_idx = np.argmin(pbest_values)
                if pbest_values[min_idx] < gbest_value:
                    gbest = pbest[min_idx]
                    gbest_value = pbest_values[min_idx]
                
                # Adaptive parameter adjustments
                self.w *= 0.99
                self.c1 *= 0.98
                self.c2 *= 1.02

            return gbest

        def de():
            # DE initialization
            population = np.array([create_particle() for _ in range(self.swarm_size)])
            for _ in range(self.budget):
                for i in range(self.swarm_size):
                    a, b, c = np.random.choice(self.swarm_size, 3, replace=False)
                    mutant = population[a] + self.mutation_factor * (population[b] - population[c])
                    crossover = np.random.rand(self.dim) < self.crossover_prob
                    trial = np.where(crossover, mutant, population[i])
                    if cost_function(trial) < cost_function(population[i]):
                        population[i] = trial
                        
                    # Adaptive parameter adjustments
                    self.mutation_factor *= 0.99
                    self.crossover_prob *= 0.98

            return population[np.argmin([cost_function(p) for p in population])]

        return pso() if np.random.rand() < 0.5 else de()