import numpy as np

class Adaptive_Quantum_Cultural_Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.belief_space = np.zeros((2, self.dim))  # Min and max belief for each dimension
        self.mutation_scale = 0.1
        self.inertia = 0.5
        self.q_factor = 0.9
        self.crossover_rate = 0.6

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.array([func(p) for p in personal_best_position])
        global_best_position = personal_best_position[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        self.belief_space[0] = lb
        self.belief_space[1] = ub

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocity[i] = (self.inertia * velocity[i] +
                               r1 * (personal_best_position[i] - position[i]) +
                               r2 * (global_best_position - position[i]))

                quantum_jump = self.q_factor * np.random.normal(scale=self.mutation_scale, size=self.dim)
                position[i] += velocity[i] + quantum_jump
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

            # Adaptive Belief Space Update
            new_belief = self.update_belief_space(personal_best_position)
            self.belief_space = new_belief

            # Crossover with belief space influence
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

    def update_belief_space(self, personal_best_position):
        # Update belief with a simple method based on the best positions
        new_belief = np.zeros_like(self.belief_space)
        new_belief[0] = np.min(personal_best_position, axis=0)
        new_belief[1] = np.max(personal_best_position, axis=0)
        return new_belief