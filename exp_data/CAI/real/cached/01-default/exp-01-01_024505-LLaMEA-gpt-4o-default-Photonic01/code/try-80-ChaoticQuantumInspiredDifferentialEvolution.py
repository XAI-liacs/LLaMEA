import numpy as np

class ChaoticQuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_agents = max(10, min(50, budget // 10))
        self.population = None
        self.mutation_factor = 0.5
        self.recombination_prob = 0.7
        self.chaos_sequence = self._generate_chaos_sequence(self.budget)
        self.interference_prob = 0.1

    def _generate_chaos_sequence(self, length):
        seq = np.zeros(length)
        x = 0.7  # initial condition for the logistic map
        r = 3.7  # chaotic parameter
        for i in range(length):
            x = r * x * (1 - x)
            seq[i] = x
        return seq

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.num_agents, self.dim)

    def evaluate_population(self, func):
        fitness = np.array([func(agent) for agent in self.population])
        return fitness

    def mutate_and_crossover(self, lb, ub, index, fitness):
        indices = [i for i in range(self.num_agents) if i != index]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant_vector = a + self.mutation_factor * (b - c)
        
        chaos_factor = self.chaos_sequence[index % len(self.chaos_sequence)]
        mutant_vector = chaos_factor * mutant_vector + (1 - chaos_factor) * self.population[index]
        
        trial_vector = np.copy(self.population[index])
        for j in range(self.dim):
            if np.random.rand() < self.recombination_prob:
                trial_vector[j] = mutant_vector[j]
        
        trial_vector = np.clip(trial_vector, lb, ub)
        trial_fitness = func(trial_vector)
        if trial_fitness < fitness[index]:
            self.population[index] = trial_vector
            fitness[index] = trial_fitness

    def apply_quantum_interference(self, lb, ub):
        for i in range(self.num_agents):
            if np.random.rand() < self.interference_prob:
                interference_vector = lb + (ub - lb) * np.random.rand(self.dim)
                self.population[i] = np.mean([self.population[i], interference_vector], axis=0)
                self.population[i] = np.clip(self.population[i], lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        fitness = self.evaluate_population(func)
        evaluations = self.num_agents

        best_idx = np.argmin(fitness)
        global_best_position = self.population[best_idx]
        global_best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            for i in range(self.num_agents):
                self.mutate_and_crossover(lb, ub, i, fitness)
            self.apply_quantum_interference(lb, ub)

            current_best_idx = np.argmin(fitness)
            current_best_fitness = fitness[current_best_idx]

            if current_best_fitness < global_best_fitness:
                global_best_fitness = current_best_fitness
                global_best_position = self.population[current_best_idx]

            evaluations += self.num_agents

        return global_best_position, global_best_fitness