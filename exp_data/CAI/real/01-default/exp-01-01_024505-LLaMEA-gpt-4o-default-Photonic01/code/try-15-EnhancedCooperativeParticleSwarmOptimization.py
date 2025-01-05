import numpy as np

class EnhancedCooperativeParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.mutation_prob = 0.1  # Probability of mutation

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        velocity = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        pop_position = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        personal_best_positions = np.copy(pop_position)
        personal_best_fitness = np.full(self.pop_size, float('inf'))

        evaluations = 0
        turbulence_intensity = 0.1  # Initial turbulence intensity

        while evaluations < self.budget:
            social_network = self._create_social_network()

            for i in range(self.pop_size):
                fitness = func(pop_position[i])
                evaluations += 1

                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = pop_position[i]

                if fitness < self.best_global_fitness:
                    self.best_global_fitness = fitness
                    self.best_global_position = pop_position[i]

                if evaluations >= self.budget:
                    break

            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # Select a social leader from the network
                social_leader = self._select_social_leader(i, social_network, personal_best_positions)

                # Update velocity
                velocity[i] = (self.inertia_weight * velocity[i] +
                               self.cognitive_coeff * r1 * (personal_best_positions[i] - pop_position[i]) +
                               self.social_coeff * r2 * (social_leader - pop_position[i]))

                # Add turbulence
                if np.random.rand() < turbulence_intensity:
                    velocity[i] += np.random.normal(0, 0.1, self.dim)

                # Apply mutation
                if np.random.rand() < self.mutation_prob:
                    mutation_vector = np.random.normal(0, 0.1, self.dim)
                    pop_position[i] = pop_position[i] + mutation_vector

                # Update position
                pop_position[i] = pop_position[i] + velocity[i]
                pop_position[i] = np.clip(pop_position[i], lb, ub)

            # Adaptive turbulence reduction
            turbulence_intensity = max(0.01, turbulence_intensity * 0.99)

        return self.best_global_position, self.best_global_fitness

    def _create_social_network(self):
        # Create a random social network with connections
        social_network = [np.random.choice(range(self.pop_size), size=3, replace=False)
                          for _ in range(self.pop_size)]
        return social_network

    def _select_social_leader(self, index, social_network, personal_best_positions):
        # Select the leader based on the network connectivity
        connected_indices = social_network[index]
        leader_fitness = float('inf')
        leader_position = None
        for idx in connected_indices:
            if personal_best_fitness[idx] < leader_fitness:
                leader_fitness = personal_best_fitness[idx]
                leader_position = personal_best_positions[idx]
        return leader_position