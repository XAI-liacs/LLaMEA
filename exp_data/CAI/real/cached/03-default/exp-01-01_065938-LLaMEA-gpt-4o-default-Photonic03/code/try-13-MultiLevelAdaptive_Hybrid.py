import numpy as np

class MultiLevelAdaptive_Hybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0
        self.c2 = 2.0
        self.inertia_max = 0.9
        self.inertia_min = 0.4
        self.mutation_rate = 0.1
        self.mutation_strength = 0.1
        self.velocity = np.zeros((self.population_size, dim))
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        best_global = pop[np.argmin(fitness)]
        best_personal = pop.copy()
        best_personal_fitness = fitness.copy()

        evaluations = self.population_size
        inertia = self.inertia_max
        temperature = 1.0  # Initial temperature for simulated annealing

        while evaluations < self.budget:
            # Update inertia
            inertia = self.inertia_max - ((self.inertia_max - self.inertia_min) * (evaluations / self.budget))
            
            # Update velocities and positions (PSO Step)
            r1, r2 = np.random.rand(2)
            self.velocity = (inertia * self.velocity +
                             self.c1 * r1 * (best_personal - pop) +
                             self.c2 * r2 * (best_global - pop))
            pop += self.velocity
            pop = np.clip(pop, lb, ub)

            # Evaluate the new population
            fitness = np.array([func(x) for x in pop])
            evaluations += self.population_size

            # Update personal and global best
            update_mask = fitness < best_personal_fitness
            best_personal[update_mask] = pop[update_mask]
            best_personal_fitness[update_mask] = fitness[update_mask]

            # Simulated Annealing based acceptance
            for i in range(self.population_size):
                delta = fitness[i] - func(best_global)
                if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
                    best_global = pop[i]
            
            # Adaptive mutation based on fitness improvement
            avg_fitness = np.mean(fitness)
            if np.min(fitness) < avg_fitness:
                self.mutation_rate = max(0.01, self.mutation_rate * 0.9)
                self.mutation_strength *= 0.9
            else:
                self.mutation_rate = min(0.2, self.mutation_rate * 1.1)
                self.mutation_strength *= 1.1

            # Apply genetic algorithm operators
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    pop[i] += np.random.normal(0, self.mutation_strength, self.dim)
                    pop[i] = np.clip(pop[i], lb, ub)

            # Cooling schedule for simulated annealing
            temperature *= 0.99

            # Save the history of best solutions
            self.history.append(best_global)

        return best_global