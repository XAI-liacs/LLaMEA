import numpy as np

class QIGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.02  # Rotation angle for quantum gate
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        q_population = np.random.uniform(0, np.pi, (self.population_size, self.dim))
        evaluations = 0

        def decode(q_individual):
            return lb + (ub - lb) * (np.sin(q_individual) ** 2)
        
        def evaluate_population(q_pop):
            return np.array([func(decode(q_individual)) for q_individual in q_pop])
        
        fitness = evaluate_population(q_population)
        best_idx = np.argmin(fitness)
        best_global = decode(q_population[best_idx])

        evaluations += self.population_size

        while evaluations < self.budget:
            next_q_population = np.zeros_like(q_population)
            for i in range(self.population_size):
                indices = np.random.choice(range(self.population_size), 2, replace=False)
                parent1, parent2 = q_population[indices]

                # Quantum crossover (average of quantum states)
                child = (parent1 + parent2) / 2
                
                # Quantum mutation (rotation gate)
                rotation = self.alpha * (np.random.rand(self.dim) - 0.5)
                child = (child + rotation) % np.pi
                
                # Decode and evaluate
                decoded_child = decode(child)
                child_fitness = func(decoded_child)
                evaluations += 1

                # Selection: replace if better
                if child_fitness < fitness[i]:
                    next_q_population[i] = child
                    fitness[i] = child_fitness
                    if child_fitness < fitness[best_idx]:
                        best_idx = i
                        best_global = decoded_child
                else:
                    next_q_population[i] = q_population[i]

            q_population = next_q_population
            self.history.append(best_global)

        return best_global