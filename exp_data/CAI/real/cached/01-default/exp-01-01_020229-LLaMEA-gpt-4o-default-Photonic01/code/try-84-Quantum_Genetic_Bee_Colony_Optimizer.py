import numpy as np

class Quantum_Genetic_Bee_Colony_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Larger initial population for diversity
        self.exploration_factor = 0.5
        self.genetic_crossover_rate = 0.8
        self.mutation_rate = 0.1
        self.q_population = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.global_best_value = float('inf')
        self.scout_bee_rate = 0.2

    def quantum_position(self, q):
        # Quantum-inspired position transformation
        return np.random.uniform(-1, 1, self.dim) * q

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        position = np.array([self.quantum_position(q) for q in self.q_population])
        position = np.clip(position, lb, ub)
        personal_best_position = np.copy(position)
        personal_best_value = np.array([func(p) for p in personal_best_position])
        global_best_position = personal_best_position[np.argmin(personal_best_value)]
        self.global_best_value = np.min(personal_best_value)
        
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Genetic-inspired crossover and mutation
                if np.random.rand() < self.genetic_crossover_rate:
                    partner_index = np.random.randint(self.population_size)
                    crossover_point = np.random.randint(self.dim)
                    new_position = np.concatenate((position[i][:crossover_point], 
                                                   position[partner_index][crossover_point:]))
                    if np.random.rand() < self.mutation_rate:
                        mutation_vector = np.random.uniform(-self.exploration_factor, self.exploration_factor, self.dim)
                        new_position += mutation_vector
                    new_position = np.clip(new_position, lb, ub)
                else:
                    new_position = position[i]
                
                current_value = func(new_position)
                evaluations += 1

                if current_value < personal_best_value[i]:
                    personal_best_position[i] = new_position
                    personal_best_value[i] = current_value

                if current_value < self.global_best_value:
                    global_best_position = new_position
                    self.global_best_value = current_value

                if evaluations >= self.budget:
                    break
            
            # Scout bee behavior for random exploration
            if np.random.rand() < self.scout_bee_rate:
                random_index = np.random.randint(self.population_size)
                new_position = np.random.uniform(lb, ub, self.dim)
                current_value = func(new_position)
                evaluations += 1

                if current_value < personal_best_value[random_index]:
                    personal_best_position[random_index] = new_position
                    personal_best_value[random_index] = current_value
                
                if current_value < self.global_best_value:
                    global_best_position = new_position
                    self.global_best_value = current_value

        return global_best_position, self.global_best_value