import numpy as np

class HybridQuantumCuckooSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.pa = 0.25  # Discovery rate of alien eggs/solutions
        self.quantum_factor_initial = 0.3
        self.quantum_factor_final = 0.1

    def levy_flight(self, L):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1/beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / abs(v) ** (1/beta)
        return L * step

    def quantum_update(self, position, global_best, eval_count):
        delta = np.random.rand(self.dim)
        lambda_factor = (eval_count / self.budget)  # Adaptive quantum factor
        quantum_factor = self.quantum_factor_initial * (1 - lambda_factor) + self.quantum_factor_final * lambda_factor
        new_position = position + quantum_factor * (global_best - position) * delta
        return new_position

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in pop])
        global_best = pop[np.argmin(fitness)]
        global_best_value = fitness.min()

        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                new_solution = pop[i] + self.levy_flight(0.01)
                new_solution = np.clip(new_solution, bounds[:, 0], bounds[:, 1])
                new_fitness = func(new_solution)
                eval_count += 1
                
                if new_fitness < fitness[i]:
                    pop[i] = new_solution
                    fitness[i] = new_fitness
                    if new_fitness < global_best_value:
                        global_best = new_solution
                        global_best_value = new_fitness

                if eval_count >= self.budget:
                    break

            # Quantum-inspired update
            for i in range(self.population_size):
                if np.random.rand() < self.pa:
                    quantum_solution = self.quantum_update(pop[i], global_best, eval_count)
                    quantum_solution = np.clip(quantum_solution, bounds[:, 0], bounds[:, 1])
                    quantum_fitness = func(quantum_solution)
                    eval_count += 1

                    if quantum_fitness < fitness[i]:
                        pop[i] = quantum_solution
                        fitness[i] = quantum_fitness
                        if quantum_fitness < global_best_value:
                            global_best = quantum_solution
                            global_best_value = quantum_fitness

                if eval_count >= self.budget:
                    break

        return global_best