import numpy as np
from collections import deque

class AGQPSO:
    def __init__(self, budget, dim, population_size=20, crossover_prob=0.8, mutation_prob=0.1, inertia=0.5, cognitive=1.5, social=1.5, quantum_prob=0.2):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.quantum_prob = quantum_prob
        self.evaluations = 0
        self.tabu_list = deque(maxlen=5)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')
        
        particles = self.initialize_particles(lb, ub)
        velocities = self.initialize_velocities()

        while self.evaluations < self.budget:
            new_particles = []
            for i in range(self.population_size):
                position = particles[i]
                
                if np.random.rand() < self.quantum_prob:
                    position = self.quantum_perturbation(position, lb, ub)

                velocities[i] = (self.inertia * velocities[i] +
                                 self.cognitive * np.random.random(self.dim) * (particles[i] - position) +
                                 self.social * np.random.random(self.dim) * (best_global_position - position if best_global_position is not None else 0))
                position = np.clip(position + velocities[i], lb, ub)
                particles[i] = position

                if tuple(position) in self.tabu_list:
                    continue

                value = func(position)
                self.evaluations += 1
                self.tabu_list.append(tuple(position))

                if value < best_global_value:
                    best_global_value = value
                    best_global_position = position

                new_particles.append(position)

                if self.evaluations >= self.budget:
                    break

            # Apply crossover and mutation
            for i in range(0, self.population_size, 2):
                if i + 1 < self.population_size and np.random.rand() < self.crossover_prob:
                    new_particles[i], new_particles[i+1] = self.crossover(new_particles[i], new_particles[i+1], lb, ub)
                if np.random.rand() < self.mutation_prob:
                    new_particles[i] = self.mutate(new_particles[i], lb, ub)
            particles = new_particles

        return best_global_position

    def initialize_particles(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def initialize_velocities(self):
        return np.random.uniform(-1, 1, (self.population_size, self.dim))

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
        return np.clip(q_position, lb, ub)

    def crossover(self, parent1, parent2, lb, ub):
        alpha = np.random.rand(self.dim)
        offspring1 = np.clip(alpha * parent1 + (1 - alpha) * parent2, lb, ub)
        offspring2 = np.clip(alpha * parent2 + (1 - alpha) * parent1, lb, ub)
        return offspring1, offspring2

    def mutate(self, individual, lb, ub):
        mutation_vector = (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
        return np.clip(individual + mutation_vector, lb, ub)