import numpy as np

class DynamicPopSizeAdaptationEHPSOLevyOptimization:
    def __init__(self, budget, dim, pop_size=30, c1=1.496, c2=1.496, w=0.729, alpha=1.5, beta=1.5, mutation_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.alpha = alpha
        self.beta = beta
        self.mutation_prob = mutation_prob

    def __call__(self, func):
        def levy_flight():
            return np.random.standard_cauchy(self.dim) / (np.random.gamma(self.alpha, 1/self.beta, self.dim) ** (1/self.beta))

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        pbest = population.copy()
        pbest_vals = np.array([func(ind) for ind in pbest])
        gbest = pbest[np.argmin(pbest_vals)]
        gbest_val = np.min(pbest_vals)

        for _ in range(self.budget):
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            velocities = self.w * velocities + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (gbest - population)
            population += velocities

            for i in range(self.pop_size):
                if np.any(population[i] < -5.0) or np.any(population[i] > 5.0):
                    population[i] = np.clip(population[i], -5.0, 5.0)

            for i in range(self.pop_size):
                if func(population[i]) < pbest_vals[i]:
                    pbest[i] = population[i]
                    pbest_vals[i] = func(population[i])

            new_gbest_val = np.min(pbest_vals)
            if new_gbest_val < gbest_val:
                gbest = pbest[np.argmin(pbest_vals)]
                gbest_val = new_gbest_val

            # Adaptive mutation rates
            update_mutation_rates = np.random.binomial(1, self.mutation_prob, self.pop_size)
            for i in range(self.pop_size):
                if update_mutation_rates[i]:
                    population[i] += 0.01 * levy_flight()

            # Dynamic population size adaptation
            if np.random.rand() < 0.2:
                self.pop_size = int(np.clip(self.pop_size + np.random.normal(0, 1), 5, 100))
                population = np.vstack([population, np.random.uniform(-5.0, 5.0, (self.pop_size - len(population), self.dim))])
                velocities = np.vstack([velocities, np.zeros((self.pop_size - len(velocities), self.dim)])
                pbest = np.vstack([pbest, population[-(self.pop_size - len(pbest)):]])
                pbest_vals = np.append(pbest_vals, [func(ind) for ind in population[-(self.pop_size - len(pbest)):]])

        return gbest