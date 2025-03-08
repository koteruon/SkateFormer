import sys

sys.path.insert(0, "")
sys.path.extend(["../"])

import numpy as np

from graph import tools

right_arm = np.array([7, 8, 22, 23]) - 1
left_arm = np.array([11, 12, 24, 25]) - 1
right_leg = np.array([13, 14, 15, 16]) - 1
left_leg = np.array([17, 18, 19, 20]) - 1
h_torso = np.array([5, 9, 6, 10]) - 1
w_torso = np.array([2, 3, 1, 4]) - 1
new_idx = np.concatenate((right_arm, left_arm, right_leg, left_leg, h_torso, w_torso), axis=-1)
original_to_new_idx = {original_idx: new_idx for new_idx, original_idx in enumerate(new_idx)}

num_node = 24
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [
    (1, 2),
    (2, 3),
    (4, 3),
    # (5, 3),
    (6, 5),
    (7, 6),
    (8, 7),
    # (9, 3),
    (10, 9),
    (11, 10),
    (12, 11),
    (13, 1),
    (14, 13),
    (15, 14),
    (16, 15),
    (17, 1),
    (18, 17),
    (19, 18),
    (20, 19),
    (22, 23),
    (23, 8),
    (24, 25),
    (25, 12),
]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
inward_new_index = [(original_to_new_idx.get(i, i), original_to_new_idx.get(j, j)) for (i, j) in inward]
outward = [(j, i) for (i, j) in inward]
outward_new_index = [(original_to_new_idx.get(i, i), original_to_new_idx.get(j, j)) for (i, j) in outward]
neighbor = inward_new_index + outward_new_index


class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    graph = AdjMatrixGraph()
    A, A_binary, A_binary_with_I = graph.A, graph.A_binary, graph.A_binary_with_I
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(A_binary_with_I, cmap="gray")
    ax[1].imshow(A_binary, cmap="gray")
    ax[2].imshow(A, cmap="gray")
    plt.show()
    print(A_binary_with_I.shape, A_binary.shape, A.shape)
