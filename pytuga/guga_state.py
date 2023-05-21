from pytuga import ShavittGraphAB, ShavittGraph
from quimb.tensor import PTensor, Tensor, TensorNetwork
import quimb.tensor as qtn
from autoray import do

class TUGAStateAB(TensorNetwork):

    def __init__(self, shavitt_graph: ShavittGraphAB, type = 'bra', params: dict = None, **tn_opts):

        self.shavitt_graph = shavitt_graph
        self.params = params

        # Build the tensor network
        tn = self.ptensor_network_state(type)

        super().__init__(tn, **tn_opts)

    @property
    def df(self):
        return self._tensor_df

    def ptensor_network_state(self,type='bra') -> TensorNetwork: # Gate ride of type as we hope Tensor.H Handles this
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
                        k_node = self.shavitt_graph.distinct_row_table[f'k{k}'][j] #TODO there seems to be a bug in quimb where these are not connected when graphng. Hopefully contraction is still good.
                        # j_tensor.new_ind(f'{j}_{k_node}', size=4)
                        inds_k.append(f'{j}_{k_node}')
                j_k_tensor =  qtn.COPY_tensor(2,inds=inds_k)

            dim_walk_ind_d = 0 # Must be a better way of doing this.
            for k in [1,2]:
                if self.shavitt_graph.distinct_row_table[f'k{k}'][j] != 'None':
                    dim_walk_ind_d += 1 

            #Parameterised tensor
            if type == 'bra':
                j_ind = f'{j}'
            else:
                j_ind = f'{j}dash'

            def elements(params):
                def matrix_slice_0(theta):
                    c = do('cos', theta)
                    s = do('sin', theta)
                    d = ((c, s,), (0., 0.))
                    return d
                def matrix_slice_1(theta):
                    c = do('cos', theta)
                    s = do('sin', theta)
                    d = ((0., 0.), (s, c))
                    return d
                data = [matrix_slice_0(params[0]), matrix_slice_1(params[1])]
                return do('array', data)

            # Not sure about this one
            def elements_row(params): # Do we need a parameter here?, Does it need to be normalised? Like in tensor networks code
                def matrix_slice(theta):
                    c = do('cos', theta / 2)
                    s = do('sin', theta / 2)
                    d = (c, -s,)
                    return d
                data = [matrix_slice(i) for i in params]
                return do('array', data)


                #This are the initial parameters
            if inds_k != [] and inds_l != []: # Middle nodes
                angles = [0,0]
                d_tensor = qtn.PTensor(elements, angles, inds=[f'{j}l',f'{j}k',j_ind])
                node_tensors.append(d_tensor)
                j_tensor = j_l_tensor & d_tensor & j_k_tensor
                # rank 2 array
            elif inds_l == []: # No upper walks. Head node. head node should be a column. But not sure if this is applicable in tensor networks as long as the indexing is correct
                # Row vector
                angles = [0]
                d_tensor = qtn.PTensor(elements_row, angles, inds=[f'{j}k',j_ind]) # Head node should be a column, BUG IT SEEMS TO BE NOT CONNECTING
                node_tensors.append(d_tensor)
                j_tensor = d_tensor & j_k_tensor
            elif inds_k == []: # No upper walks. Tail node
                # column vector
                angles = [0]
                d_tensor = qtn.PTensor(elements_row, angles, inds=[f'{j}l',j_ind])
                j_tensor = j_l_tensor & d_tensor
                node_tensors.append(d_tensor)
            else:
                raise(ValueError('BUG ALERT'))

            j_tensors.append(j_tensor)

        self.shavitt_graph.j_tensors = j_tensors
        
        tensor_df = self.shavitt_graph.distinct_row_table.copy()
        self._tensor_df = tensor_df.assign(tensor=node_tensors)

        # Build tensor network
        tensor_network_state = j_tensors[0]
        for j in j_tensors[1:]:
            tensor_network_state = j & tensor_network_state

        if type == 'bra':
            tensor_network_state = tensor_network_state.H

        return tensor_network_state