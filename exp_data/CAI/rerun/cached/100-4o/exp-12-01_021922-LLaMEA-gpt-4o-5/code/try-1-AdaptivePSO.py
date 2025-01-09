import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim, swarm_size=50, inertia_max=0.9, inertia_min=0.4):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia_max = inertia_max
        self.inertia_min = inertia_min
        self.position = np.random.uniform(-5, 5, (self.swarm_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.personal_best_position = np.copy(self.position)
        self.personal_best_value = np.full(self.swarm_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.c1 = 2.0
        self.c2 = 2.0
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.swarm_size):
                fitness_value = func(self.position[i])
                self.evaluations += 1
                if fitness_value < self.personal_best_value[i]:
                    self.personal_best_value[i] = fitness_value
                    self.personal_best_position[i] = self.position[i]
                if fitness_value < self.global_best_value:
                    self.global_best_value = fitness_value
                    self.global_best_position = self.position[i]
            
            # Stochastic inertia weight adjustment
            inertia_weight = np.random.uniform(self.inertia_min, self.inertia_max)
            
            for i in range(self.swarm_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    self.c1 * r1 * (self.personal_best_position[i] - self.position[i]) +
                                    self.c2 * r2 * (self.global_best_position - self.position[i]))
                self.position[i] += self.velocity[i]
                self.position[i] = np.clip(self.position[i], -5, 5)
                
            # Local Search Intensification
            if np.random.rand() < 0.1:
                random_idx = np.random.randint(0, self.swarm_size)
                local_search_position = self.position[random_idx] + np.random.uniform(-0.1, 0.1, self.dim)
                local_search_position = np.clip(local_search_position, -5, 5)
                local_fitness = func(local_search_position)
                self.evaluations += 1
                if local_fitness < self.personal_best_value[random_idx]:
                    self.personal_best_value[random_idx] = local_fitness
                    self.personal_best_position[random_idx] = local_search_position
                if local_fitness < self.global_best_value:
                    self.global_best_value = local_fitness
                    self.global_best_position = local_search_position

        return self.global_best_value, self.global_best_position