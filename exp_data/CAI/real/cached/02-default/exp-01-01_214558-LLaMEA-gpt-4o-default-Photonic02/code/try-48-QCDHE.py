import numpy as np

class QCDHE:
    def __init__(self, budget, dim, num_nodes=5, node_size=20, inertia=0.5, cognitive=1.4, social=1.4, quantum_prob=0.2, hyperedge_prob=0.3, cooperation_intensity=0.6):
        self.budget = budget
        self.dim = dim
        self.num_nodes = num_nodes
        self.node_size = node_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.quantum_prob = quantum_prob
        self.hyperedge_prob = hyperedge_prob
        self.cooperation_intensity = cooperation_intensity
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')

        nodes = [self.initialize_node(lb, ub) for _ in range(self.num_nodes)]
        velocities = [np.random.uniform(-1, 1, (self.node_size, self.dim)) for _ in range(self.num_nodes)]

        while self.evaluations < self.budget:
            self.dynamic_hypergraph_adaptation(nodes)

            for node_id in range(self.num_nodes):
                local_best_position = None
                local_best_value = float('inf')
                
                for i in range(self.node_size):
                    position = nodes[node_id][i]
                    
                    if np.random.rand() < self.quantum_prob:
                        position = self.quantum_perturbation(position, lb, ub)
                    
                    velocities[node_id][i] = (self.inertia * velocities[node_id][i] +
                                              self.cognitive * np.random.random(self.dim) * (nodes[node_id][i] - position) +
                                              self.social * np.random.random(self.dim) * (best_global_position - position if best_global_position is not None else 0))
                    position = np.clip(position + velocities[node_id][i], lb, ub)
                    nodes[node_id][i] = position

                    value = func(position)
                    self.evaluations += 1

                    if value < local_best_value:
                        local_best_value = value
                        local_best_position = position

                    if self.evaluations >= self.budget:
                        break

                if local_best_value < best_global_value:
                    best_global_value = local_best_value
                    best_global_position = local_best_position

                if self.evaluations >= self.budget:
                    break

        return best_global_position

    def initialize_node(self, lb, ub):
        return np.random.uniform(lb, ub, (self.node_size, self.dim))

    def quantum_perturbation(self, position, lb, ub):
        scale = 0.1
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * scale
        return np.clip(q_position, lb, ub)

    def dynamic_hypergraph_adaptation(self, nodes):
        for i in range(len(nodes)):
            if np.random.rand() < self.hyperedge_prob:
                hyper_edge = np.random.choice(len(nodes), size=int(self.num_nodes * self.cooperation_intensity), replace=False)
                for h_node in hyper_edge:
                    recipient = nodes[i][np.random.randint(0, self.node_size)]
                    donor = nodes[h_node][np.random.randint(0, self.node_size)]
                    crossover_point = np.random.randint(1, self.dim)
                    recipient[:crossover_point] = donor[:crossover_point].copy()