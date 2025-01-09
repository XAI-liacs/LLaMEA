import numpy as np

class ImprovedHybridPSODE(HybridPSODE):
    def __call__(self, func):
        def initialize_velocity():
            return np.zeros((self.population_size, self.dim))

        def update_velocity(position, velocity, pbest, gbest, w=0.5, c1=1.5, c2=1.5):
            r1, r2 = np.random.rand(2)
            new_velocity = w * velocity + c1 * r1 * (pbest - position) + c2 * r2 * (gbest - position)
            return new_velocity

        def optimize():
            population = initialize_population()
            velocity = initialize_velocity()
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