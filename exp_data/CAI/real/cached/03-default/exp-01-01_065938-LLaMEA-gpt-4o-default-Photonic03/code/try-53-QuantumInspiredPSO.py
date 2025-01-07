import numpy as np

class QuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter
        self.w = 0.5   # Inertia weight
        self.diversity_factor = 0.1
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        p_best = np.copy(pop)
        p_best_fitness = np.array([func(x) for x in pop])
        g_best_idx = np.argmin(p_best_fitness)
        g_best = pop[g_best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                rp = np.random.rand(self.dim)
                rg = np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * rp * (p_best[i] - pop[i]) +
                                 self.c2 * rg * (g_best - pop[i]))

                quantum_noise = np.random.uniform(-self.diversity_factor, self.diversity_factor, self.dim)
                pop[i] += velocities[i] + quantum_noise
                pop[i] = np.clip(pop[i], lb, ub)

                fitness = func(pop[i])
                evaluations += 1

                if fitness < p_best_fitness[i]:
                    p_best[i] = pop[i]
                    p_best_fitness[i] = fitness

                if fitness < p_best_fitness[g_best_idx]:
                    g_best_idx = i
                    g_best = pop[i]

            self.history.append(g_best)
            if evaluations >= self.budget:
                break

            # Update inertia weight and diversity factor dynamically
            self.w = np.clip(self.w + 0.05 * (np.random.rand() - 0.5), 0.4, 0.9)
            self.diversity_factor = np.clip(self.diversity_factor + 0.02 * (np.random.rand() - 0.5), 0.05, 0.2)

        return g_best