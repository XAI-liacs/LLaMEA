import numpy as np

class QuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0  # Cognitive constant
        self.c2 = 2.0  # Social constant
        self.inertia_max = 0.9
        self.inertia_min = 0.4
        self.velocity = np.zeros((self.population_size, dim))
        self.history = []
        self.quantum_prob = 0.05  # Probability of applying quantum jump

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        best_global = pop[np.argmin(fitness)]
        best_personal = pop.copy()
        best_personal_fitness = fitness.copy()

        evaluations = self.population_size
        inertia = self.inertia_max

        while evaluations < self.budget:
            inertia = self.inertia_max - ((self.inertia_max - self.inertia_min) * (evaluations / self.budget))
            
            r1, r2 = np.random.rand(2)
            self.velocity = (inertia * self.velocity +
                             self.c1 * r1 * (best_personal - pop) +
                             self.c2 * r2 * (best_global - pop))
            pop += self.velocity
            pop = np.clip(pop, lb, ub)

            # Quantum-inspired jump
            quantum_jump_mask = np.random.rand(self.population_size) < self.quantum_prob
            for idx in np.where(quantum_jump_mask)[0]:
                pop[idx] = best_global + np.random.normal(0, 0.1, self.dim)
                pop[idx] = np.clip(pop[idx], lb, ub)

            fitness = np.array([func(x) for x in pop])
            evaluations += self.population_size

            update_mask = fitness < best_personal_fitness
            best_personal[update_mask] = pop[update_mask]
            best_personal_fitness[update_mask] = fitness[update_mask]

            if np.min(fitness) < func(best_global):
                best_global = pop[np.argmin(fitness)]

            # Adaptive quantum probability based on entropy of fitness distribution
            fitness_entropy = -np.sum((fitness / np.sum(fitness)) * np.log(fitness / np.sum(fitness) + 1e-9))
            self.quantum_prob = 0.05 + 0.1 * (1 - fitness_entropy / np.log(self.population_size))

            self.history.append(best_global)

        return best_global