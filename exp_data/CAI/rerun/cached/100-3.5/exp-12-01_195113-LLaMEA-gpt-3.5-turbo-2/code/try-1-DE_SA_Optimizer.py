import numpy as np

class DE_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(10, dim*2)  # Adaptive population size based on dimensionality
        self.F = 0.5  # Differential Evolution parameter
        self.CR = 0.9  # Differential Evolution parameter
        self.T_init = 1.0  # Initial temperature for Simulated Annealing
        self.alpha = 0.9  # Cooling rate for Simulated Annealing

    def __call__(self, func):
        def mutate(x, pop):
            indices = np.random.choice(len(pop), 3, replace=False)
            a, b, c = pop[indices]
            mutant = a + self.F * (b - c)
            return np.clip(mutant, -5.0, 5.0)

        def simulated_annealing(curr, candidate, T):
            delta_E = func(candidate) - func(curr)
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
                return candidate
            else:
                return curr

        def optimize():
            population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
            best_solution = population[np.argmin([func(ind) for ind in population])]
            T = self.T_init
            for _ in range(self.budget - self.pop_size):
                new_population = []
                for ind in population:
                    candidate = mutate(ind, population)
                    candidate = simulated_annealing(ind, candidate, T)
                    new_population.append(candidate)
                    if func(candidate) < func(best_solution):
                        best_solution = candidate
                population = np.array(new_population)
                T *= self.alpha
            return best_solution

        return optimize()