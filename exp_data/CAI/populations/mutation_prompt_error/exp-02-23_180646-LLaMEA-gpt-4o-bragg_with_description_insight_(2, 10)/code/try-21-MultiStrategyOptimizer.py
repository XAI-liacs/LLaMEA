import numpy as np
from scipy.optimize import minimize

class MultiStrategyOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = None
        self.ub = None

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        population_size = 12 + 3 * self.dim
        F, CR = 0.5, 0.9
        population = self.initialize_population(population_size)
        fitness = np.array([func(ind) for ind in population])
        eval_count = population_size

        while eval_count < self.budget:
            # Adaptive DE loop
            for i in range(population_size):
                if eval_count >= self.budget:
                    break

                # DE mutation with adaptive F
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                F = np.random.uniform(0.4, 0.7)  # Adaptive F for exploration
                mutant = np.clip(a + F * (b - c), self.lb, self.ub)

                # Enforce periodicity
                mutant = self.enforce_periodicity(mutant)

                # DE crossover with adaptive CR
                CR = np.random.uniform(0.7, 0.95)  # Adaptive CR for diversity
                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])

                # Evaluate trial solution
                trial_fitness = func(trial)
                eval_count += 1

                # Selection process
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Simulated Annealing for enhanced local refinement
            if eval_count < self.budget:
                current_sol = population[np.argmin(fitness)]
                current_fitness = np.min(fitness)
                temperature = 1.0
                cooling_rate = 0.95

                for _ in range(5):  # Simulated Annealing steps
                    if eval_count >= self.budget:
                        break
                    neighbor = current_sol + np.random.uniform(-0.1, 0.1, self.dim)
                    neighbor = np.clip(neighbor, self.lb, self.ub)
                    neighbor_fitness = func(neighbor)
                    eval_count += 1
                    acceptance_prob = np.exp((current_fitness - neighbor_fitness) / temperature)
                    if neighbor_fitness < current_fitness or np.random.rand() < acceptance_prob:
                        current_sol = neighbor
                        current_fitness = neighbor_fitness
                    temperature *= cooling_rate

                # Update best found solution
                if current_fitness < np.min(fitness):
                    best_idx = np.argmin(fitness)
                    population[best_idx] = current_sol
                    fitness[best_idx] = current_fitness

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def initialize_population(self, size):
        return np.random.uniform(self.lb, self.ub, (size, self.dim))

    def enforce_periodicity(self, vector):
        # Self-adaptive periodicity control
        period = np.random.randint(2, 5)
        num_periods = self.dim // period
        for i in range(num_periods):
            mean_value = np.mean(vector[i*period:(i+1)*period])
            vector[i*period:(i+1)*period] = mean_value
        return vector