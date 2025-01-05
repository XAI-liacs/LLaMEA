import numpy as np

class QuantumInspiredMultiSwarmPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 3
        self.swarm_size = 20
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.quantum_influence = 0.3

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarms = [np.random.uniform(lb, ub, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        pbest_positions = [swarm.copy() for swarm in swarms]
        pbest_fitness = [np.array([func(p) for p in swarm]) for swarm in swarms]
        gbest_fitness = [np.min(pbf) for pbf in pbest_fitness]
        gbest_positions = [swarm[np.argmin(pbf)] for swarm, pbf in zip(swarms, pbest_fitness)]

        evaluations = self.num_swarms * self.swarm_size

        while evaluations < self.budget:
            for swarm_idx in range(self.num_swarms):
                swarm = swarms[swarm_idx]
                velocity = velocities[swarm_idx]
                pbest_pos = pbest_positions[swarm_idx]
                gbest_pos = gbest_positions[swarm_idx]

                r1 = np.random.rand(self.swarm_size, self.dim)
                r2 = np.random.rand(self.swarm_size, self.dim)
                velocity = (self.inertia_weight * velocity +
                            self.cognitive_coeff * r1 * (pbest_pos - swarm) +
                            self.social_coeff * r2 * (gbest_pos - swarm))

                quantum_jump = np.random.uniform(-self.quantum_influence, self.quantum_influence, (self.swarm_size, self.dim))
                velocity += quantum_jump

                swarm += velocity
                swarm = np.clip(swarm, lb, ub)

                fitness = np.array([func(p) for p in swarm])
                evaluations += self.swarm_size

                improved = fitness < pbest_fitness[swarm_idx]
                pbest_positions[swarm_idx][improved] = swarm[improved]
                pbest_fitness[swarm_idx][improved] = fitness[improved]

                if np.min(fitness) < gbest_fitness[swarm_idx]:
                    gbest_fitness[swarm_idx] = np.min(fitness)
                    gbest_positions[swarm_idx] = swarm[np.argmin(fitness)]

            # Cooperative behavior by exchanging gbest among swarms
            for i in range(self.num_swarms):
                for j in range(i + 1, self.num_swarms):
                    if gbest_fitness[i] < gbest_fitness[j]:
                        gbest_positions[j] = gbest_positions[i]
                        gbest_fitness[j] = gbest_fitness[i]
                    elif gbest_fitness[j] < gbest_fitness[i]:
                        gbest_positions[i] = gbest_positions[j]
                        gbest_fitness[i] = gbest_fitness[j]

        # Return the best global solution found across all swarms
        global_best_idx = np.argmin(gbest_fitness)
        return gbest_positions[global_best_idx]