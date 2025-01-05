import numpy as np

class QuantumHybridSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.f = 0.5
        self.cr = 0.7
        self.w_min = 0.2
        self.w_max = 0.8
        self.alpha = 0.9  # Quantum step size
        self.position = None
        self.velocity = None
        self.pbest = None
        self.pbest_scores = None
        self.gbest = None
        self.gbest_score = float('inf')
        self.chaotic_sequence = self.generate_advanced_chaotic_sequence()
        self.global_best_improvement = 0.01

    def generate_advanced_chaotic_sequence(self, length=10000):
        sequence = np.zeros(length)
        sequence[0] = 0.6
        for i in range(1, length):
            sequence[i] = 4.0 * sequence[i-1] * (1 - sequence[i-1]) ** 2
        return sequence

    def chaotic_parameter(self, index):
        return self.chaotic_sequence[index % len(self.chaotic_sequence)]

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.velocity = np.random.rand(self.population_size, self.dim) - 0.5
        self.pbest = np.copy(self.position)
        self.pbest_scores = np.full(self.population_size, float('inf'))

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.population_size):
            if scores[i] < self.pbest_scores[i]:
                self.pbest_scores[i] = scores[i]
                self.pbest[i] = self.position[i]
            if scores[i] < self.gbest_score:
                self.gbest_score = scores[i]
                self.gbest = self.position[i]
        return scores

    def update_inertia_weight(self, iteration, max_iterations):
        return self.w_max - ((self.w_max - self.w_min) * (iteration / max_iterations))

    def update_velocity_position(self, iteration, max_iterations):
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        chaotic_factor = self.chaotic_parameter(iteration)
        inertia_weight = self.update_inertia_weight(iteration, max_iterations)
        cognitive = chaotic_factor * self.c1 * r1 * (self.pbest - self.position)
        social = chaotic_factor * self.c2 * r2 * (self.gbest - self.position)
        self.velocity = inertia_weight * self.velocity + cognitive + social
        self.position += self.velocity

    def quantum_behavior_update(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        quantum_move = np.random.uniform(-1, 1, (self.population_size, self.dim))
        quantum_step = self.alpha * quantum_move * (self.gbest - self.position)
        quantum_position = self.position + quantum_step
        quantum_position = np.clip(quantum_position, lb, ub)
        return quantum_position

    def differential_evolution_mutation(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        new_population = np.copy(self.position)
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = self.pbest[a] + self.f * (self.pbest[b] - self.pbest[c])
            mutant = np.clip(mutant, lb, ub)
            crossover = np.random.rand(self.dim) < self.cr
            if not np.any(crossover):
                crossover[np.random.randint(0, self.dim)] = True
            new_population[i] = np.where(crossover, mutant, self.position[i])
        return new_population

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        iteration = 0
        last_best_score = float('inf')
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            score_improvement = np.abs(self.gbest_score - last_best_score)
            if score_improvement < self.global_best_improvement:
                self.w_max *= 0.9  # Adaptive strategy based on improvement
            
            self.update_velocity_position(iteration, max_iterations)
            new_population = self.differential_evolution_mutation(func.bounds)
            quantum_population = self.quantum_behavior_update(func.bounds)
            self.position = np.where(
                scores[:, np.newaxis] <= np.array([func(p) for p in new_population])[:, np.newaxis],
                self.position, new_population
            )
            self.position = np.where(
                scores[:, np.newaxis] <= np.array([func(p) for p in quantum_population])[:, np.newaxis],
                self.position, quantum_population
            )
            func_calls += self.population_size * 2
            last_best_score = self.gbest_score
            iteration += 1
        return self.gbest, self.gbest_score