import numpy as np

class HybridSwarmEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = max(20, min(60, budget // 8))
        self.particles = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.velocities = None
        self.global_best_position = None
        self.inertia = 0.7
        self.cognitive_param = 1.5
        self.social_param = 1.5
        self.mutation_factor = 0.8
        self.crossover_probability = 0.7

    def initialize_swarm(self, lb, ub):
        self.particles = lb + (ub - lb) * np.random.rand(self.swarm_size, self.dim)
        self.velocities = np.zeros((self.swarm_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_fitnesses = np.full(self.swarm_size, float('inf'))

    def evaluate_swarm(self, func):
        fitness = np.array([func(p) for p in self.particles])
        for i in range(self.swarm_size):
            if fitness[i] < self.personal_best_fitnesses[i]:
                self.personal_best_fitnesses[i] = fitness[i]
                self.personal_best_positions[i] = self.particles[i]
        global_best_index = np.argmin(self.personal_best_fitnesses)
        if self.personal_best_fitnesses[global_best_index] < self.best_fitness:
            self.best_fitness = self.personal_best_fitnesses[global_best_index]
            self.global_best_position = self.personal_best_positions[global_best_index]

    def update_velocities_and_positions(self, lb, ub):
        r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
        cognitive_component = self.cognitive_param * r1 * (self.personal_best_positions - self.particles)
        social_component = self.social_param * r2 * (self.global_best_position - self.particles)
        self.velocities = self.inertia * self.velocities + cognitive_component + social_component
        self.particles += self.velocities
        self.particles = np.clip(self.particles, lb, ub)

    def differential_evolution(self, lb, ub):
        for i in range(self.swarm_size):
            idxs = [idx for idx in range(self.swarm_size) if idx != i]
            a, b, c = self.particles[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
            crossover = np.random.rand(self.dim) < self.crossover_probability
            self.particles[i] = np.where(crossover, mutant, self.particles[i])

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_swarm(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_swarm(func)
            evaluations += self.swarm_size
            if evaluations >= self.budget:
                break
            self.update_velocities_and_positions(lb, ub)
            self.differential_evolution(lb, ub)

        return self.global_best_position, self.best_fitness