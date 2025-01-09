import numpy as np

class PSOGWO:
    def __init__(self, budget, dim, swarm_size=20, alpha=0.1, beta=0.1, delta=0.1):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def __call__(self, func):
        def fitness(position):
            return func(position)

        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))

        def update_position(position, velocity):
            return np.clip(position + velocity, -5.0, 5.0)

        def pso_update_position(pos, vel, pbest, gbest):
            r1, r2 = np.random.rand(2)
            new_vel = self.alpha * vel + self.beta * r1 * (pbest - pos) + self.delta * r2 * (gbest - pos)
            new_pos = update_position(pos, new_vel)
            return new_pos, new_vel

        def gwo_update_position(alpha, beta, delta, position):
            a, b, c = position
            r1, r2 = np.random.rand(2)
            A1 = 2 * r1 - 1
            A2 = 2 * r2 - 1
            D_alpha = abs(self.alpha * alpha - a)
            D_beta = abs(self.beta * beta - b)
            D_delta = abs(self.delta * delta - c)
            new_pos = np.clip(alpha - self.alpha * A1 * D_alpha, -5.0, 5.0)
            new_pos = np.clip(new_pos + beta - self.beta * A2 * D_beta, -5.0, 5.0)
            new_pos = np.clip(new_pos + delta - self.delta * A2 * D_delta, -5.0, 5.0)
            return new_pos

        population = initialize_population()
        velocities = np.zeros((self.swarm_size, self.dim))
        best_positions = population.copy()
        best_fitness = np.array([fitness(pos) for pos in best_positions])
        gbest_idx = np.argmin(best_fitness)
        gbest = best_positions[gbest_idx]

        for _ in range(self.budget):
            for i in range(self.swarm_size):
                pos = population[i]
                vel = velocities[i]

                # PSO update
                new_pos, new_vel = pso_update_position(pos, vel, best_positions[i], gbest)
                population[i] = new_pos
                velocities[i] = new_vel

                # GWO update
                new_pos = gwo_update_position(gbest, best_positions[i], best_positions[gbest_idx], new_pos)
                population[i] = new_pos

                # Update best positions
                curr_fitness = fitness(new_pos)
                if curr_fitness < best_fitness[i]:
                    best_positions[i] = new_pos
                    best_fitness[i] = curr_fitness
                    if curr_fitness < best_fitness[gbest_idx]:
                        gbest_idx = i
                        gbest = new_pos

        return gbest