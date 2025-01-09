import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.max_iter = budget // self.population_size
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

        def clipToBounds(population):
            return np.clip(population, self.lb, self.ub)

        def evaluate_population(population):
            return np.array([func(individual) for individual in population])

        def update_velocity(position, velocity, pbest, gbest, w=0.5, c1=1.5, c2=1.5):
            r1, r2 = np.random.rand(2)
            new_velocity = w * velocity + c1 * r1 * (pbest - position) + c2 * r2 * (gbest - position)
            return new_velocity

        def optimize():
            population = initialize_population()
            population = clipToBounds(population)
            fitness = evaluate_population(population)
            pbest = population.copy()
            gbest = population[np.argmin(fitness)]

            for _ in range(self.max_iter):
                for i in range(self.population_size):
                    new_velocity = update_velocity(population[i], velocity[i], pbest[i], gbest)
                    new_position = population[i] + new_velocity
                    new_position = np.clip(new_position, self.lb, self.ub)
                    if func(new_position) < func(population[i]):
                        population[i] = new_position
                    if func(new_position) < func(pbest[i]):
                        pbest[i] = new_position
                    if func(new_position) < func(gbest):
                        gbest = new_position

            return gbest

        return optimize()