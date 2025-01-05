import numpy as np

class PSO_QIM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.w = 0.5  # inertia weight
        self.c1 = 1.5 # cognitive coefficient
        self.c2 = 1.5 # social coefficient
        self.mutation_prob = 0.1
        self.history = []

    def quantum_mutation(self, particle, lb, ub):
        alpha = np.random.uniform(0, 1, self.dim)
        beta = np.random.uniform(0, 1, self.dim)
        mutated = lb + (ub - lb) * alpha * np.sin(2 * np.pi * beta)
        return np.clip(mutated, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = np.copy(pop)
        personal_best_fitness = np.array([func(x) for x in pop])
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_idx]
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - pop[i]) +
                                 self.c2 * r2 * (global_best - pop[i]))
                pop[i] += velocities[i]
                pop[i] = np.clip(pop[i], lb, ub)

                if np.random.rand() < self.mutation_prob:
                    pop[i] = self.quantum_mutation(pop[i], lb, ub)

                fitness = func(pop[i])
                evaluations += 1

                if fitness < personal_best_fitness[i]:
                    personal_best[i] = pop[i]
                    personal_best_fitness[i] = fitness

            global_best_idx = np.argmin(personal_best_fitness)
            global_best = personal_best[global_best_idx]
            self.history.append(global_best)

        return global_best