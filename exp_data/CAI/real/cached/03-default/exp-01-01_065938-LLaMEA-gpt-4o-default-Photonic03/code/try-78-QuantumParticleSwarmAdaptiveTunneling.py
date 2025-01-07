import numpy as np

class QuantumParticleSwarmAdaptiveTunneling:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1, self.c2 = 1.5, 1.5
        self.w_min, self.w_max = 0.4, 0.9
        self.tunneling_prob = 0.1
        self.history = []

    def adaptive_tunneling(self, position, best_global, lb, ub):
        tunnel = np.random.uniform(-0.1, 0.1, self.dim)
        if np.random.rand() < self.tunneling_prob:
            new_position = position + tunnel * (best_global - position)
            return np.clip(new_position, lb, ub)
        return position

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        best_individual = pop.copy()
        best_individual_fitness = fitness.copy()
        best_global_idx = np.argmin(fitness)
        best_global = pop[best_global_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)

            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (w * velocities[i] +
                                 self.c1 * r1 * (best_individual[i] - pop[i]) +
                                 self.c2 * r2 * (best_global - pop[i]))
                pop[i] = np.clip(pop[i] + velocities[i], lb, ub)
                pop[i] = self.adaptive_tunneling(pop[i], best_global, lb, ub)

                current_fitness = func(pop[i])
                evaluations += 1

                if current_fitness < best_individual_fitness[i]:
                    best_individual[i] = pop[i]
                    best_individual_fitness[i] = current_fitness
                    if current_fitness < fitness[best_global_idx]:
                        best_global_idx = i
                        best_global = pop[i]

            self.history.append(best_global)

        return best_global