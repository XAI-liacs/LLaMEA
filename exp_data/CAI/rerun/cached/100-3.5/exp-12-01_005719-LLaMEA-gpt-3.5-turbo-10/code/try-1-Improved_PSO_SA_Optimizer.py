import numpy as np

class Improved_PSO_SA_Optimizer(PSO_SA_Optimizer):
    def __call__(self, func):
        def improved_pso_sa_optimize():
            # PSO initialization
            swarm_position = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
            swarm_velocity = np.zeros((self.swarm_size, self.dim))
            pbest_position = swarm_position.copy()
            pbest_value = np.full(self.swarm_size, np.inf)
            gbest_position = np.zeros(self.dim)
            gbest_value = np.inf

            # SA initialization
            temperature = self.max_temp

            for _ in range(self.budget):
                # PSO update
                for i in range(self.swarm_size):
                    fitness = func(swarm_position[i])
                    if fitness < pbest_value[i]:
                        pbest_value[i] = fitness
                        pbest_position[i] = swarm_position[i].copy()
                    if fitness < gbest_value:
                        gbest_value = fitness
                        gbest_position = swarm_position[i].copy()

                    r1, r2 = np.random.rand(), np.random.rand()
                    inertia_weight = self.w * (1.0 - _ / self.budget)  # Dynamic inertia weight adaptation
                    swarm_velocity[i] = inertia_weight * swarm_velocity[i] + self.c1 * r1 * (pbest_position[i] - swarm_position[i]) + self.c2 * r2 * (gbest_position - swarm_position[i])
                    swarm_position[i] += swarm_velocity[i]
                    swarm_position[i] = np.clip(swarm_position[i], -5.0, 5.0)

                # SA update
                for i in range(self.swarm_size):
                    new_position = pbest_position[i] + np.random.uniform(-1, 1, self.dim) * temperature
                    new_position = np.clip(new_position, -5.0, 5.0)
                    new_fitness = func(new_position)
                    current_fitness = func(pbest_position[i])
                    if new_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - new_fitness) / temperature):
                        pbest_position[i] = new_position
                        pbest_value[i] = new_fitness

                temperature *= self.cooling_rate

            return gbest_position

        return improved_pso_sa_optimize()