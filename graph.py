import maxflow
import numpy as np
import sys
from tqdm import tqdm
import cv2

def disparity(image_left, image_right, **kwargs):
    solver = GraphCutDisparitySolver(image_left, image_right, **kwargs)
    return solver.solve()

def to_gray(image):
    if len(image.shape) == 2:
        return image.astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

# Based on https://github.com/pmonasse/disparity-with-graph-cuts
class GraphCutDisparitySolver:
    LABEL_OCCLUDED = 1

    NODE_ALPHA = -1
    NODE_ABSENT = -2

    IS_NODE = lambda x: x >=0

    def __init__(
        self,
        image_left,
        image_right,
        always_randomize=False,
        search_depth=30,
        max_levels=-1,
        max_iterations=4,
        occlusion_cost=-1,
        smoothness_cost_high=-1,
        smoothness_cost_low=-1,
        smoothness_threshold=8,
        census_kernel_size = 7,
        occlusion_cost_ratio = -1,
        dissim_method = 'census'
    ):

        self.isGray = (len(image_left.shape) == 2) or (image_left[:,:,0] == image_left[:,:,1]).all()
        self.image_left = image_left
        self.image_right = image_right
        self.image_shape = self.image_left.shape[:2]
        self.image_size = self.image_left[:,:,0].size
        self.image_indices = np.indices(self.image_shape)
        self.census_kernel_size = census_kernel_size
        self.census_IL = self.census_transform(self.image_left, KW=self.census_kernel_size)
        self.census_IR = self.census_transform(self.image_right, KW=self.census_kernel_size)
        self.energy = float('inf')
        self.dissim_method = dissim_method

        self.always_randomize = always_randomize
        self.search_depth = search_depth
        self.max_levels = self.search_depth if max_levels < 0 else max_levels
        self.max_iterations = max_iterations
        self.occlusion_cost_ratio = 0.25  if occlusion_cost_ratio < 0 else occlusion_cost_ratio
        self.occlusion_cost = occlusion_cost if occlusion_cost > 0 else self.compute_k()
        self.smoothness_cost_low = smoothness_cost_low if smoothness_cost_low > 0 else 0.2 * self.occlusion_cost
        self.smoothness_cost_high = smoothness_cost_high if smoothness_cost_high > 0 else 3 * self.smoothness_cost_low
        self.smoothness_threshold = smoothness_threshold


        search_interval = (self.search_depth // self.max_levels) + bool(self.search_depth % self.max_levels)
        self.search_levels = -1 * np.arange(0, self.search_depth + 1, search_interval)[::-1]
        rank = np.empty(len(self.search_levels), dtype=int)
        rank[np.argsort(self.search_levels)] = np.arange(len(self.search_levels))
        self.label_rank = dict(zip(self.search_levels, rank))

        self.build_neighbors()

    def is_in_image(self, x):
        return (0 <= x) & (x < self.image_shape[1])

    def build_neighbors(self):
        indices = np.indices(self.image_shape)

        neighbors_one_p = indices[:, 1:, :].reshape(2, -1)
        neighbors_one_q = neighbors_one_p + [[-1],[0]]
        neighbors_two_p = indices[:, :, :-1].reshape(2, -1)
        neighbors_two_q = neighbors_two_p + [[0],[1]]

        self.neighbors = np.array([
            np.concatenate([neighbors_one_p, neighbors_two_p], axis=1),
            np.concatenate([neighbors_one_q, neighbors_two_q], axis=1),
        ])
        self.neighbors_rolled = list(np.rollaxis(self.neighbors, 1))

        indices_p, indices_q = self.neighbors
        #diff_left = self.image_left[list(indices_p)] - self.image_left[list(indices_q)]
        diff_left = self.dissim_neighbor(list(indices_p), list(indices_q), leftright='l', method=self.dissim_method)

        self.is_left_under = np.abs(diff_left) < self.smoothness_threshold
        #self.is_left_under = (diff_left) < self.smoothness_threshold

    def solve(self):
        self.labels = np.full(self.image_shape, self.LABEL_OCCLUDED, dtype=np.int)
        label_done = np.zeros(len(self.search_levels), dtype=bool)

        for i in tqdm(range(self.max_iterations)):
            if i == 0 or self.always_randomize:
                label_order = np.random.permutation(self.search_levels)

            for label in tqdm(label_order):
                # print('iteration', i, 'label', label)
                label_index = self.label_rank[label]
                if label_done[label_index]:
                    continue

                is_expanded = self.expand_label(label)
                if is_expanded:
                    label_done[:] = False
                label_done[label_index] = True


            if label_done.all():
                break


        return -1 * self.labels

    def expand_label(self, label):
        is_expanded = False
        g = maxflow.Graph[int](2*self.image_size, 12*self.image_size)
        # print('Adding data+occlusion terms for', label)
        self.add_data_occlusion_terms(g, label)
        # print('Adding smoothness terms for', label)
        self.add_smoothness_terms(g, label)
        # print('Adding uniqueness terms for', label)
        self.add_uniqueness_terms(g, label)

        energy = g.maxflow() + self.e_data_occlusion
        if energy < self.energy:
            # print('new energy', energy, 'updating labels', label)
            self.update_labels(g, label)
            is_expanded = True
        self.energy = energy
        return is_expanded

    def dissim(self, indices_y, indices_shifted, method='census'):

        if method == 'census':
            return  np.sum(np.bitwise_xor(self.census_IL, self.census_IR[indices_y,indices_shifted]), axis=2).astype(np.float32)
        elif method == 'ssd':
            return np.sum(np.square(self.image_left - self.image_right[indices_y,indices_shifted]).astype(np.float32),axis=2)

    def dissim_neighbor(self, p_idx, q_idx, leftright ,method='census'):
        if method == 'census':
            if leftright == 'l':
                return np.sum(np.bitwise_xor(self.census_IL[p_idx], self.census_IL[q_idx]), axis=1).astype(np.float32)
            elif leftright == 'r':
                return np.sum(np.bitwise_xor(self.census_IR[p_idx], self.census_IR[q_idx]), axis=1).astype(np.float32)
        elif method == 'ssd':
            if leftright == 'l':
                return np.sum(abs(self.image_left[p_idx] - self.image_left[q_idx]).astype(np.float32),axis=1)
            elif leftright == 'r':
                return np.sum(abs(self.image_right[p_idx] - self.image_right[q_idx]).astype(np.float32), axis=1)


    def add_data_occlusion_terms(self, g, label):
        indices_y, indices_x = self.image_indices
        is_label = self.labels == label
        is_occluded = self.labels == self.LABEL_OCCLUDED

        indices_shifted = np.where(is_occluded, indices_x, indices_x + self.labels)
        assert self.is_in_image(indices_shifted[np.logical_not(is_occluded)]).all()


        # Replace with Census Cost
        ssd_active =self.dissim(indices_y, indices_shifted, method=self.dissim_method)  - self.occlusion_cost
        ssd_active[is_occluded | is_label] = -self.occlusion_cost - 1
        nodes_active = np.zeros(self.image_shape, dtype=np.int)
        nodes_active[is_occluded] = self.NODE_ABSENT
        nodes_active[is_label] = self.NODE_ALPHA
        is_node_active = np.logical_not(is_label | is_occluded)
        e_data_occlusion = ssd_active[is_label].sum()

        is_occluded = np.logical_not(self.is_in_image(indices_x + label))
        indices_shifted = np.where(is_occluded, indices_x, indices_x + label)

        # Replace with Census Cost
        ssd_label = self.dissim(indices_y, indices_shifted, method=self.dissim_method)  - self.occlusion_cost

        ssd_label[is_occluded | is_label] = -self.occlusion_cost - 1
        nodes_label = np.zeros(self.image_shape, dtype=np.int)
        nodes_label[is_occluded] = self.NODE_ABSENT
        nodes_label[is_label] = self.NODE_ALPHA
        is_node_label = np.logical_not(is_label | is_occluded)

        num_nodes = is_node_label.sum() + is_node_active.sum()
        node_ids = g.add_nodes(num_nodes)
        
        if is_node_active.sum() != 0:
            g.add_grid_tedges(node_ids[-is_node_active.sum():], 0, ssd_active[is_node_active])
            mask_active = np.nonzero(is_node_active)
            nodes_active[mask_active] = node_ids[-is_node_active.sum():]

        if is_node_label.sum() != 0:
            g.add_grid_tedges(node_ids[:is_node_label.sum()], ssd_label[is_node_label], 0)
            mask_label = np.nonzero(is_node_label)
            nodes_label[mask_label] = node_ids[:is_node_label.sum()]


        self.is_node_active = is_node_active
        self.is_node_label = is_node_label
        self.nodes_active = nodes_active
        self.nodes_label = nodes_label
        self.e_data_occlusion = e_data_occlusion

    def add_smoothness_terms(self, g, label):
        labels_p, labels_q = self.labels[self.neighbors_rolled]

        penalty_label = self.get_smoothness_penalty(label)
        penalty_active_p = self.get_smoothness_penalty(labels_p)
        penalty_active_q = self.get_smoothness_penalty(labels_q)

        indices_p, indices_q = self.neighbors
        is_p_in_range = self.is_in_image(indices_p[1, :] + labels_q)
        is_q_in_range = self.is_in_image(indices_q[1, :] + labels_p)

        indice_y_array,  indice_x_array= self.neighbors.T[:,0,:], self.neighbors.T[:,1,:]
        label_p_array, label_q_array =  self.labels[indice_y_array[:,0], indice_x_array[:,0]], self.labels[indice_y_array[:,1], indice_x_array[:,1]]
        node_l_p_array, node_l_q_array =  self.nodes_label[indice_y_array[:,0], indice_x_array[:,0]], self.nodes_label[indice_y_array[:,1], indice_x_array[:,1]]
        node_a_p_array, node_a_q_array = self.nodes_active[indice_y_array[:, 0], indice_x_array[:, 0]], self.nodes_active[indice_y_array[:, 1], indice_x_array[:, 1]]
        is_p_active_array, is_q_active_array = self.is_node_active[indice_y_array[:, 0], indice_x_array[:, 0]], self.is_node_active[indice_y_array[:, 1], indice_x_array[:, 1]]

        node_l_not_absent = np.logical_and(node_l_p_array !=  self.NODE_ABSENT,  node_l_q_array !=  self.NODE_ABSENT)
        node_lpq_not_absentalpha = np.logical_and(node_l_not_absent,np.logical_and(node_l_p_array !=  self.NODE_ALPHA,  node_l_q_array !=  self.NODE_ALPHA))
        penalty_lpq_not_absentalph = penalty_label[node_lpq_not_absentalpha]
        self.add_smoothness_weights_vectorized(g,node_l_p_array[node_lpq_not_absentalpha],node_l_q_array[node_lpq_not_absentalpha],np.zeros(len(penalty_lpq_not_absentalph)),penalty_lpq_not_absentalph,penalty_lpq_not_absentalph,np.zeros(len(penalty_lpq_not_absentalph)))

        node_lp_not_absentalpha = np.logical_and(node_l_not_absent,np.logical_and(node_l_p_array !=  self.NODE_ALPHA,  node_l_q_array ==  self.NODE_ALPHA))
        penalty_lp_not_absentalph = penalty_label[node_lp_not_absentalpha]
        self.add_tedges_vectorized(g, node_l_p_array[node_lp_not_absentalpha], np.zeros(len(penalty_lp_not_absentalph)), penalty_lp_not_absentalph)


        node_lq_not_absentalpha = np.logical_and(node_l_not_absent, np.logical_and(node_l_p_array == self.NODE_ALPHA, node_l_q_array != self.NODE_ALPHA))
        penalty_lq_not_absentalph = penalty_label[node_lq_not_absentalpha]
        self.add_tedges_vectorized(g, node_l_p_array[node_lq_not_absentalpha], np.zeros(len(penalty_lq_not_absentalph)), penalty_lq_not_absentalph)

        label_p_eq_q = label_p_array == label_q_array
        label_p_eq_q_both_active = np.logical_and(label_p_eq_q,np.logical_and(is_p_active_array,is_q_active_array))

        penalty_active_p_p_eq_q_both_active = penalty_active_p[label_p_eq_q_both_active]
        self.add_smoothness_weights_vectorized(g, node_a_p_array[label_p_eq_q_both_active], node_a_q_array[label_p_eq_q_both_active], np.zeros(len(penalty_active_p_p_eq_q_both_active)), penalty_active_p_p_eq_q_both_active, penalty_active_p_p_eq_q_both_active, np.zeros(len(penalty_active_p_p_eq_q_both_active)))

        label_p_neq_q = label_p_array != label_q_array
        p_active_p_in_range = np.logical_and(is_p_active_array,is_q_in_range)
        mask = np.logical_and(label_p_neq_q,p_active_p_in_range)
        self.add_tedges_vectorized(g, node_a_p_array[mask], np.zeros(len(penalty_active_p[mask])), penalty_active_p[mask])

        q_active_p_in_range = np.logical_and(is_q_active_array, is_p_in_range)
        mask = np.logical_and(label_p_neq_q, q_active_p_in_range)
        self.add_tedges_vectorized(g, node_a_q_array[mask], np.zeros(len(penalty_active_q[mask])), penalty_active_q[mask])


    def _shift(self, indices, shift):
        _, width = self.image_shape
        indices_shifted = np.copy(indices)
        indices_shifted[1, :] += shift
        is_in_image = self.is_in_image(indices_shifted[1, :])
        indices_shifted[1, :] = np.clip(indices_shifted[1, :], 0, width - 1)
        return indices_shifted, is_in_image

    def get_smoothness_penalty(self, labels):
        indices_p, indices_q = self.neighbors
        if type(labels) is np.ndarray:
            labels = labels[self.is_left_under]

        smoothness = np.full(indices_p.shape[1], self.smoothness_cost_low, dtype=np.float)

        indices_p_shifted, is_p_in_image = self._shift(indices_p[:, self.is_left_under], labels)
        indices_q_shifted, is_q_in_image = self._shift(indices_q[:, self.is_left_under], labels)
        #diff_right = self.image_right[list(indices_p_shifted)] - self.image_right[list(indices_q_shifted)]
        diff_right = self.dissim_neighbor(list(indices_p_shifted), list(indices_q_shifted), leftright='r', method=self.dissim_method)


        is_left_under = np.copy(self.is_left_under)
        is_left_under[is_left_under] = np.abs(diff_right) < self.smoothness_threshold
        smoothness[is_left_under] = self.smoothness_cost_high

        is_left_under[:] = self.is_left_under
        is_left_under[is_left_under] = np.logical_not(is_p_in_image & is_q_in_image)
        smoothness[is_left_under] = 0

        return smoothness

    def add_smoothness_weights(self, g, node1, node2, w1, w2, w3, w4):
        w0 = w1 - w2
        g.add_tedge(node1, w4, w2)
        g.add_tedge(node2, 0, w0)
        g.add_edge(node1, node2, 0, w3 - w4 - w0)
    
    def add_smoothness_weights_vectorized(self, g, node1, node2, w1, w2, w3, w4):
        w0 = w1 - w2
        if node1.any():
            g.add_grid_tedges(node1, w4, w2)
        if node2.any():
            g.add_grid_tedges(node2, np.zeros(len(node2)), w0)
        if node1.any() and node2.any():
            g.add_edges(node1, node2, np.zeros(len(node2)), w3 - w4 - w0)

    def add_tedges_vectorized(self, g, nodes, cap, rcap):
        if nodes.any():
            g.add_grid_tedges(nodes, cap, rcap)


    def add_uniqueness_terms(self, g, label):
        # assert (self.labels[self.is_node_active] != self.LABEL_OCCLUDED).all()

        _, width = self.image_shape
        indices_y, indices_x = self.image_indices
        indices_shifted = indices_x + self.labels - label
        is_shift_valid = self.is_in_image(indices_shifted)
        indices_shifted = np.clip(indices_shifted, 0, width - 1)
        forbid = self.is_node_active & is_shift_valid
        forbid_label = self.nodes_label[indices_y, indices_shifted][forbid]
        forbid_active = self.nodes_active[forbid]
        self.add_uniqueness_weights(g, forbid_active, forbid_label)
        # assert (forbid_label >= 0).all()
        # assert (forbid_active >= 0).all()

        is_node_label = self.nodes_label != self.NODE_ABSENT
        forbid = self.is_node_active & is_node_label
        self.add_uniqueness_weights(g, self.nodes_active[forbid], self.nodes_label[forbid])
        # assert (self.nodes_label[forbid] >= 0).all()
        # assert (self.nodes_active[forbid] >= 0).all()

    def add_uniqueness_weights(self, g, sources, targets):
        if sources.any():
            g.add_edges(sources, targets, np.full(len(sources),sys.maxsize), np.zeros(len(sources)))

    def update_labels(self, g, label):
        is_node_active = np.copy(self.is_node_active)
        if is_node_active.any():
            nodes_active = self.nodes_active[is_node_active]
            is_node_active[is_node_active] = g.get_grid_segments(nodes_active)
            self.labels[is_node_active] = self.LABEL_OCCLUDED

        is_node_label = np.copy(self.is_node_label)
        if is_node_label.any():
            nodes_label = self.nodes_label[is_node_label]
            is_node_label[is_node_label] = g.get_grid_segments(nodes_label)
            self.labels[is_node_label] = label

    def census_transform(self, src, KW=3):
        h, w, cs = src.shape

        pad = KW // 2
        src = cv2.copyMakeBorder(src, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT_101)

        h_pad, w_pad, cs = src.shape

        census = np.zeros((h, w, cs, KW ** 2 - 1), dtype=np.bool)
        center_pixels = src[pad:h_pad - pad, pad:w_pad - pad]

        offsets = [(u, v) for v in range(KW) for u in range(KW) if not u == pad == v]
        for i, (u, v) in enumerate(offsets):
            census[:, :, :, i] = src[v:v + h_pad - (KW - 1), u:u + w_pad - (KW - 1)] >= center_pixels

        return census.reshape((h,w,len(offsets)*cs))


    def compute_k(self):
        K = int(self.search_depth*self.occlusion_cost_ratio)
        h, w, _ = self.image_left.shape
        L_costVol = np.ones((h, w, self.search_depth + 1))

        if self.dissim_method == 'census':
            for s in range(self.search_depth + 1):
                L_costVol[:, s:w, s] = np.sum(np.bitwise_xor(self.census_IL[:, s:w, :], self.census_IR[:, :w - s, :]), axis=2)
                if s > 0:
                    L_costVol[:, :s, s] = np.tile(np.sum(np.bitwise_xor(self.census_IL[:, s, :], self.census_IR[:, 0, :]), axis=1)[:, np.newaxis],
                                                  (1, s))
        elif self.dissim_method == 'ssd':
            for s in range(self.search_depth + 1):
                L_costVol[:, s:w, s] = np.sum(np.square(self.image_left[:, s:w, :]-self.image_right[:, :w - s, :]), axis=2)
                if s > 0:
                    L_costVol[:, :s, s] = np.tile(np.sum(np.square(self.image_left[:, s, :]-self.image_right[:, 0, :]), axis=1)[:, np.newaxis],
                                                  (1, s))
        L_costVol.sort(axis=2)
        avg_cost = int(np.average(L_costVol[:,:,K]))

        return avg_cost

