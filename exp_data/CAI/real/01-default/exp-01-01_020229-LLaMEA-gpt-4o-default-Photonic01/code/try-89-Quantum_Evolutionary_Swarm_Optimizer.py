import numpy as np

class Quantum_Evolutionary_Swarm_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Elevated to enhance diversity
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.6  # Reduced for stability
        self.q_factor = 0.8
        self.crossover_rate = 0.4
        self.mutation_scale = 0.05
        self.inertia_decay = 0.97  # Gradual decay of inertia
        self.temperature = 0.9
        
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
                self.w *= self.inertia_decay
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (global_best_position - position[i]))
                
                quantum_jump = self.q_factor * np.random.normal(scale=self.mutation_scale, size=self.dim)
                annealing_shift = np.exp(-evaluations / (self.budget * self.temperature)) * np.random.uniform(-0.5, 0.5, self.dim)
                
                position[i] += velocity[i] + quantum_jump + annealing_shift
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

            # Genetic-like Crossover operation
            for i in range(0, self.population_size, 2):
                if np.random.rand() < self.crossover_rate:
                    parent1 = personal_best_position[i]
                    parent2 = personal_best_position[(i + 1) % self.population_size]
                    offspring1, offspring2 = self.crossover(parent1, parent2)
                    
                    # Evaluate offspring and potentially update personal and global bests
                    self.evaluate_and_update(offspring1, func, personal_best_position, personal_best_value, global_best_position, global_best_value, evaluations, lb, ub)
                    self.evaluate_and_update(offspring2, func, personal_best_position, personal_best_value, global_best_position, global_best_value, evaluations, lb, ub)

        return global_best_position, global_best_value

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, self.dim - 1)
        offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return offspring1, offspring2

    def evaluate_and_update(self, position, func, personal_best_position, personal_best_value, global_best_position, global_best_value, evaluations, lb, ub):
        position = np.clip(position, lb, ub)
        current_value = func(position)
        evaluations += 1
        
        if current_value < personal_best_value[np.argmin(personal_best_value)]:
            personal_best_position[np.argmin(personal_best_value)] = position
            personal_best_value[np.argmin(personal_best_value)] = current_value
        
        if current_value < global_best_value:
            global_best_position = position
            global_best_value = current_value