import numpy as np

class Enhanced_DE_SA_Optimizer_Refined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 10 * self.dim
        CR = 0.9
        F = 0.8
        T0 = 1.0
        alpha = 0.95

        def chaotic_map_mutation(x, pop, F):
            a, b, c = pop[np.random.choice(len(pop), 3, replace=False)]
            F = np.clip(F + np.random.normal(0, 0.2), 0.2, 1.0)  # Adaptive F
            chaotic_map = lambda x: np.sin(3.9 * np.sin(3.9 * np.sin(3.9 * x)))  # Chaotic map function
            chaotic_vals = chaotic_map(np.linspace(0, 1, self.dim))
            mutant = np.where(np.random.uniform(0, 1, self.dim) < chaotic_vals, a + F * (b - c), x)
            return np.clip(mutant, -5.0, 5.0)

        def adaptive_simulated_annealing(x, best_x, T):
            new_x = x + np.random.normal(0, T, size=self.dim)
            new_x = np.clip(new_x, -5.0, 5.0)
            T = np.clip(T * 0.99, 0.1, T0)  # Adaptive T
            if func(new_x) < func(x) or np.random.rand() < np.exp((func(x) - func(new_x)) / T):
                return new_x
            else:
                return x

        def fitness_diversity_selection(population, func):
            fitness_vals = [func(x) for x in population]
            unique_fitness = len(set(fitness_vals))
            diversity_ratio = unique_fitness / len(population)
            sorted_indices = np.argsort(fitness_vals)
            selected_indices = sorted_indices[:int(0.8 * len(population))]  # Select top 80% by fitness
            return population[selected_indices]

        population = np.random.uniform(-5.0, 5.0, size=(pop_size, self.dim))
        best_x = population[np.argmin([func(x) for x in population)]
        
        for _ in range(self.budget):
            new_population = []
            T = T0 * alpha ** _
            selected_population = fitness_diversity_selection(population, func)
            for x in selected_population:
                trial_x = chaotic_map_mutation(x, selected_population, F)
                trial_x = adaptive_simulated_annealing(trial_x, best_x, T)
                new_population.append(trial_x)
                if func(trial_x) < func(best_x):
                    best_x = trial_x
            population = np.array(new_population)
        
        return best_x