import numpy as np

class DNPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.9
        self.cognitive_component = 2.0
        self.social_component = 2.0
        self.max_velocity = 0.1 * (func.bounds.ub - func.bounds.lb)
        self.min_velocity = -self.max_velocity

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(self.min_velocity, self.max_velocity, (self.population_size, self.dim))
        personal_best_positions = np.copy(pop)
        personal_best_fitness = np.array([func(x) for x in personal_best_positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_fitness)]
        
        evaluations = self.population_size

        while evaluations < self.budget:
            for i, particle in enumerate(pop):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_component * r1 * (personal_best_positions[i] - particle)
                    + self.social_component * r2 * (global_best_position - particle)
                )
                velocities[i] = np.clip(velocities[i], self.min_velocity, self.max_velocity)

                pop[i] = np.clip(particle + velocities[i], lb, ub)
                current_fitness = func(pop[i])
                evaluations += 1

                if current_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = pop[i]
                    personal_best_fitness[i] = current_fitness

                    if current_fitness < np.min(personal_best_fitness):
                        global_best_position = pop[i]

            self.inertia_weight = max(0.4, 0.9 - 0.5 * (evaluations / self.budget))
        
        return global_best_position