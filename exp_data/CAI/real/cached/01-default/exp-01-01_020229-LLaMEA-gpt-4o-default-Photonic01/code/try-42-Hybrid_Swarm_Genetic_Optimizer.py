import numpy as np

class Hybrid_Swarm_Genetic_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.crossover_rate = 0.5
        self.q_factor = 0.8
        self.mutation_prob = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.array([func(p) for p in personal_best_position])
        global_best_position = personal_best_position[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)
        
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (global_best_position - position[i]))
                
                position[i] += self.q_factor * (velocity[i] + np.random.normal(size=self.dim))
                position[i] = np.clip(position[i], lb, ub)

                current_value = func(position[i])
                evaluations += 1
                
                if current_value < personal_best_value[i]:
                    personal_best_position[i] = position[i]
                    personal_best_value[i] = current_value
                
                if current_value < global_best_value:
                    global_best_position = position[i]
                    global_best_value = current_value

                if evaluations >= self.budget:
                    break
            
            # Genetic Crossover and Mutation
            if np.random.rand() < self.crossover_rate:
                parent1, parent2 = np.random.choice(self.population_size, 2, replace=False)
                crossover_point = np.random.randint(1, self.dim)
                child = np.concatenate((personal_best_position[parent1][:crossover_point],
                                        personal_best_position[parent2][crossover_point:]))
                
                if np.random.rand() < self.mutation_prob:
                    mutation_index = np.random.randint(self.dim)
                    child[mutation_index] = np.random.uniform(lb[mutation_index], ub[mutation_index])
                
                child_value = func(child)
                evaluations += 1
                if child_value < global_best_value:
                    global_best_position = child
                    global_best_value = child_value

        return global_best_position, global_best_value