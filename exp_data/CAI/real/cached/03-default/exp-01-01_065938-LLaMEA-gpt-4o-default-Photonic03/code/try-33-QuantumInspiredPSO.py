import numpy as np

class QuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.eta = 0.1  # quantum movement factor

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-0.1, 0.1, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        personal_best = pop.copy()
        personal_best_fitness = fitness.copy()
        global_best_idx = np.argmin(fitness)
        global_best = pop[global_best_idx]
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - pop[i]) +
                                 self.c2 * r2 * (global_best - pop[i]))

                # Add quantum inspired movement
                quantum_move = self.eta * np.random.normal(size=self.dim) * np.sign(global_best - pop[i])
                new_position = pop[i] + velocities[i] + quantum_move
                new_position = np.clip(new_position, lb, ub)

                new_fitness = func(new_position)
                evaluations += 1

                if new_fitness < personal_best_fitness[i]:
                    personal_best[i] = new_position
                    personal_best_fitness[i] = new_fitness

                    if new_fitness < personal_best_fitness[global_best_idx]:
                        global_best_idx = i
                        global_best = new_position

                pop[i] = new_position

        return global_best