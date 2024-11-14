import numpy as np

class HybridPSOEnhancedCuckooSearch:
    def __init__(self, budget, dim, population_size=10, pa=0.25, alpha=0.01, elitism_rate=0.1, pso_weight=0.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.pa = pa
        self.alpha = alpha
        self.elitism_rate = elitism_rate
        self.pso_weight = pso_weight

    def levy_flight(self):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        step = u / abs(v) ** (1 / beta)
        return step

    def pso_step(self, particle, best_particle, global_best):
        inertia_weight = np.random.uniform(0.4, 0.9)
        cognitive_component = np.random.rand() * self.pso_weight * (best_particle - particle)
        social_component = np.random.rand() * self.pso_weight * (global_best - particle)
        return inertia_weight * particle + cognitive_component + social_component

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        fitness = [func(x) for x in population]
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        global_best = best_solution

        for _ in range(self.budget):
            new_population = []
            for i, cuckoo in enumerate(population):
                step_size = self.levy_flight()
                cuckoo_new = cuckoo + step_size * np.random.randn(self.dim)
                cuckoo_new = np.clip(cuckoo_new, -5.0, 5.0)

                if np.random.rand() > self.pa:
                    idx = np.random.randint(self.population_size)
                    cuckoo_new = cuckoo_new + self.alpha * (population[idx] - cuckoo_new)

                new_fitness = func(cuckoo_new)
                if new_fitness < fitness[i]:
                    population[i] = cuckoo_new
                    fitness[i] = new_fitness

                    if new_fitness < fitness[best_idx]:
                        best_idx = i
                        best_solution = cuckoo_new

            # Introduce dynamic population size adaptation
            if np.random.rand() < self.elitism_rate:
                worst_idx = np.argmax(fitness)
                population[worst_idx] = best_solution
                fitness[worst_idx] = func(best_solution)

            # PSO step
            for i, particle in enumerate(population):
                population[i] = self.pso_step(particle, population[best_idx], global_best)
                if func(population[i]) < func(global_best):
                    global_best = population[i]

        return best_solution