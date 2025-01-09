import numpy as np

class DEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.max_iter = budget // self.pop_size

    def __call__(self, func):
        def fitness(x):
            return func(x)

        def init_population():
            return np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))

        def update_velocity(position, velocity, pbest, gbest, w=0.5, c1=0.5, c2=0.5):
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            new_velocity = w * velocity + c1 * r1 * (pbest - position) + c2 * r2 * (gbest - position)
            return np.clip(new_velocity, -1, 1)

        population = init_population()
        fitness_values = np.array([fitness(ind) for ind in population])
        pbest = population.copy()
        gbest = population[np.argmin(fitness_values)]
        velocity = np.zeros((self.pop_size, self.dim))

        for _ in range(self.max_iter):
            for i, ind in enumerate(population):
                new_position = ind + velocity[i]
                new_fitness = fitness(new_position)
                if new_fitness < fitness_values[i]:
                    fitness_values[i] = new_fitness
                    pbest[i] = new_position
                    if new_fitness < fitness(gbest):
                        gbest = new_position
                velocity = update_velocity(ind, velocity, pbest, gbest)

        return gbest