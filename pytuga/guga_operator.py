from pytuga.angular_momentum import spin_coupling_tensor
from pytuga import ShavittGraphAB, ShavittGraph
from quimb.tensor import PTensor, Tensor, TensorNetwork
import quimb.tensor as qtn
from autoray import do
import numpy as np

class TUGAOperatorAB(TensorNetwork):

    def __init__(self, shavitt_graph: ShavittGraphAB, **tn_opts):

        self.shavitt_graph = shavitt_graph

        # Build the tensor network
        tn = self.tensor_network_operator()

        super().__init__(tn, **tn_opts)

    @property
    def df(self):
        return self._tensor_df

    def tensor_network_operator(self) -> TensorNetwork:  #Need to figure our what goes in the tensor
        #k are downward chaining indices
        j_tensors = []
        node_tensors = []
        # Build the node tensors
        for j in reversed(self.shavitt_graph.distinct_row_table.index.tolist()):
            # tagged [ j , u , uirrep] Maybe this could just be linked to the row of drt.
            # If ever in doubt just link index j to drt
            u = self.shavitt_graph.distinct_row_table['u'][j]
            u = f'U{u}'
            uirrep = self.shavitt_graph.distinct_row_table['uirrep'][j]
            uirrep = f'UIRREP{uirrep}'

            # Upper Walks (Very confusing l is upper)
            inds_l = [] # This is quite sketchky think of better way
            if j != self.shavitt_graph.distinct_row_table.index.tolist()[0]:
                inds_l.append(f'{j}l')
                for l in [1,2]:
                    if self.shavitt_graph.distinct_row_table[f'l{l}'][j] != 'None':
                        l_node = self.shavitt_graph.distinct_row_table[f'l{l}'][j]
                        inds_l.append(f'{l_node}_{j}')
                j_l_tensor = qtn.COPY_tensor(2,inds=inds_l) # I thinks this is 4 nodt len(inds_l)-1

            # Lower Walks
            inds_k = []
            if j != self.shavitt_graph.distinct_row_table.index.tolist()[-1]:
                inds_k.append(f'{j}k')
                for k in [1,2]:
                    if self.shavitt_graph.distinct_row_table[f'k{k}'][j] != 'None':
                        k_node = self.shavitt_graph.distinct_row_table[f'k{k}'][j]
                        # j_tensor.new_ind(f'{j}_{k_node}', size=4)
                        inds_k.append(f'{j}_{k_node}')
                j_k_tensor =  qtn.COPY_tensor(2,inds=inds_k)

            dim_walk_ind_d = 0 # Must be a better way of doing this.
            for k in [1,2]: # TODO This should work for any GUGA graph
                if self.shavitt_graph.distinct_row_table[f'k{k}'][j] != 'None':
                    dim_walk_ind_d += 1 

            spin = self.shavitt_graph.distinct_row_table.b[j] / 2
            #This are the initial parameters

            j_ind = f'{j}'
            j_dash_ind = f'{j}dash'

            # Here we need an index that carries hamiltonian term coeefficientsicc

            if inds_k != [] and inds_l != []: # Middle nodes
                d_tensor = qtn.Tensor(np.array([spin_coupling_tensor(spin)]), inds=[f'{j}l',f'{j}k',j_ind, j_dash_ind]) # 
                # print(np.array([spin_coupling_tensor(spin)]).shape) #TODO make sure the indexs match up correctly
                j_tensor = j_l_tensor & d_tensor & j_k_tensor
                # rank 2 array
            elif inds_l == []: # No upper walks. Head node. head node should be a column. But not sure if this is applicable in tensor networks as long as the indexing is correct
                # Row vector
                d_tensor = qtn.Tensor(spin_coupling_tensor(spin), inds=[f'{j}k',j_ind,j_dash_ind]) 
                j_tensor = d_tensor & j_k_tensor
            elif inds_k == []: # No upper walks. Tail node
                # column vector
                d_tensor = qtn.Tensor(spin_coupling_tensor(spin), inds=[f'{j}l',j_ind,j_dash_ind])
                j_tensor = j_l_tensor & d_tensor 
            else:
                raise(ValueError('BUG ALERT'))
            
            node_tensors.append(d_tensor)
            j_tensors.append(j_tensor)
            
        tensor_df = self.shavitt_graph.distinct_row_table.copy()
        self._tensor_df = tensor_df.assign(tensor=node_tensors)

        # Build tensor network
        tensor_network_operator = j_tensors[0]
        for j in j_tensors[1:]:
            tensor_network_operator = j & tensor_network_operator

        # Does the ket in the operator network need to be transposed?

        return tensor_network_operator