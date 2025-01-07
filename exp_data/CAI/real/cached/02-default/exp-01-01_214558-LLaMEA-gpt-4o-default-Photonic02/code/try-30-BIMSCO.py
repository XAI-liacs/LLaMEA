import numpy as np

class BIMSCO:
    def __init__(self, budget, dim, num_species=3, species_size=10, interaction_prob=0.3, mutation_rate=0.1, recombination_rate=0.2):
        self.budget = budget
        self.dim = dim
        self.num_species = num_species
        self.species_size = species_size
        self.interaction_prob = interaction_prob
        self.mutation_rate = mutation_rate
        self.recombination_rate = recombination_rate
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')

        species = [self.initialize_species(lb, ub) for _ in range(self.num_species)]

        while self.evaluations < self.budget:
            for species_id in range(self.num_species):
                current_species = species[species_id]
                other_species = [species[i] for i in range(self.num_species) if i != species_id]

                for individual_id in range(self.species_size):
                    individual = current_species[individual_id]

                    # Interaction with other species
                    if np.random.rand() < self.interaction_prob:
                        partner_species = np.random.choice(other_species)
                        partner = partner_species[np.random.randint(self.species_size)]
                        interaction = self.species_interaction(individual, partner, lb, ub)
                        current_species[individual_id] = interaction

                    # Mutation
                    if np.random.rand() < self.mutation_rate:
                        self.species_mutation(current_species[individual_id], lb, ub)

                    # Fitness evaluation
                    value = func(current_species[individual_id])
                    self.evaluations += 1

                    if value < best_global_value:
                        best_global_value = value
                        best_global_position = current_species[individual_id].copy()

                    if self.evaluations >= self.budget:
                        return best_global_position

                # Recombination within species
                self.recombine_species(current_species, lb, ub)

        return best_global_position

    def initialize_species(self, lb, ub):
        return np.random.uniform(lb, ub, (self.species_size, self.dim))

    def species_interaction(self, individual, partner, lb, ub):
        interaction_vector = (partner - individual) * np.random.rand(self.dim) * 0.1
        new_position = individual + interaction_vector
        return np.clip(new_position, lb, ub)

    def species_mutation(self, individual, lb, ub):
        mutation_vector = (np.random.rand(self.dim) - 0.5) * 0.1 * (ub - lb)
        individual += mutation_vector
        np.clip(individual, lb, ub, out=individual)

    def recombine_species(self, species, lb, ub):
        for i in range(0, self.species_size, 2):
            if i+1 < self.species_size and np.random.rand() < self.recombination_rate:
                crossover_point = np.random.randint(1, self.dim)
                species[i][:crossover_point], species[i+1][:crossover_point] = (
                    species[i+1][:crossover_point].copy(), species[i][:crossover_point].copy())