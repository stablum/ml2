#!/usr/bin/env python
import numpy as np

PRINT = False

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def normalize(vector):
    s = sum(vector)
    if s==0:
        # prevent division by 0
        return vector

    return vector/s

def my_argmax(a):
    return np.unravel_index(np.argmax(a),a.shape)

class Node(object):
    """
    Base-class for Nodes in a factor graph. Only instantiate sub-classes of Node.
    """
    def __init__(self, name):
        # A name for this Node, for printing purposes
        self.name = name
        
        # Neighbours in the graph, identified with their index in this list.
        # i.e. self.neighbours contains neighbour 0 through len(self.neighbours) - 1.
        self.neighbours = []
        
        # Reset the node-state (not the graph topology)
        self.reset()
    
    def reset(self):
        # Incoming messages; a dictionary mapping neighbours to messages.
        # That is, it maps  Node -> np.ndarray.
        self.in_msgs = {}
        
        # A set of neighbours for which this node has pending messages.
        # We use a python set object so we don't have to worry about duplicates.
        self.pending = set([])

    def add_neighbour(self, nb):
        self.neighbours.append(nb)

    def initialize_messages(self, uniform_value):
        raise Exception('Method initialize_messages not implemented in base-class Node')

    def send_sp_msg(self, other):
        # To be implemented in subclass.
        raise Exception('Method send_sp_msg not implemented in base-class Node')
   
    def send_ms_msg(self, other):
        # To be implemented in subclass.
        raise Exception('Method send_ms_msg not implemented in base-class Node')

    def can_send_to(self, node):
        return set(self.neighbours).difference(set([node])).issubset(self.in_msgs.keys())

    def set_pending_except(self,exclude=None):
        if exclude is None:
            exclude = []
        if type(exclude) is not list:
            exclude = [exclude]
        self.pending = set()
        for neigh in self.neighbours:
            if self.can_send_to(neigh):
                self.pending.add(neigh)
   
    def send_pending(self,algo='sp'):
        for node in list(self.pending):
            #print "in",self,"link to",node,"is pending. sending message..."
            if algo == 'sp':
                self.send_sp_msg(node)
            elif algo == 'ms':
                self.send_ms_msg(node)
            else:
                raise Exception("algorithm "+algo+" unknown")

    def receive_msg(self, other, msg):  
        #print self.name,"received message",msg,"from",str(other)
        # Store the incomming message, replacing previous messages from the same node
        self.in_msgs[other] = msg

        # TODO: add pending messages
        self.set_pending_except(other)

    def _in_msgs_except(self,exclude=None):
        if exclude is None:
            exclude = []
        if type(exclude) is not list:
            exclude = [exclude]
        msg_vectors = [ 
            self.in_msgs[v] 
            for v 
            in self.neighbours
            if v not in exclude
        ]
        return msg_vectors

    def _in_msgs_reduce(self,reduce_func, factor_func, exclude=None):
        msg_vectors = self._in_msgs_except(exclude)

        if len(msg_vectors) == 0:
            # leaf node?
            return factor_func(np.array(1))

        ixstuff = np.ix_(*msg_vectors)
        ret = reduce(reduce_func, ixstuff)
        return ret

    def _collapse(self, func, summand, exclude=None):
        if exclude is None:
            exclude = []
        if type(exclude) is not list:
            exclude = [exclude]
        
        sum_axes = range(len(self.neighbours))
        for curr in exclude:
            del sum_axes[curr]
        
        if len(sum_axes) == 0:
            # well, no axis to summate, this means that the input is returned unaltered
            return summand
        ret = summand
        for i in sum_axes[::-1]:
            ret = func(ret,axis=i)
        return ret

class Variable(Node):
    def __init__(self, name, num_states):
        """
        Variable node constructor.
        Args:
            name: a name string for this node. Used for printing. 
            num_states: the number of states this variable can take.
            Allowable states run from 0 through (num_states - 1).
            For example, for a binary variable num_states=2,
            and the allowable states are 0, 1.
        """
        self.num_states = num_states
        
        # Call the base-class constructor
        super(Variable, self).__init__(name)
    
    def initialize_messages(self, value):
        if type(value) is np.ndarray:
            msg = value
        else:
            msg = np.array([value] * self.num_states)
        
        for neigh in self.neighbours:
            self.in_msgs[neigh] = msg

    def set_observed(self, observed_state):
        """
        Set this variable to an observed state.
        Args:
            observed_state: an integer value in [0, self.num_states - 1].
        """
        # Observed state is represented as a 1-of-N variable
        # Could be 0.0 for sum-product, but log(0.0) = -inf so a tiny value is preferable for max-sum
        self.observed_state[:] = 0.000001
        self.observed_state[observed_state] = 1.0
        
    def set_latent(self):
        """
        Erase an observed state for this variable and consider it latent again.
        """
        # No state is preferred, so set all entries of observed_state to 1.0
        # Using this representation we need not differentiate between observed and latent
        # variables when sending messages.
        self.observed_state[:] = 1.0
     
    def is_latent(self):
        return np.array_equal(self.observed_state, np.ones(len(self.observed_state)))

    def is_observed(self):
        return not self.is_latent()

    def reset(self):
        super(Variable, self).reset()
        self.observed_state = np.ones(self.num_states)

    def marginal(self,Z=None): #for sum product
        return self.marginal_generic(np.multiply, Z)

    def marginal_ms(self, Z=None):
        return self.marginal_generic(np.add, Z)

    def marginal_generic(self, reduce_func, Z=None):
        """
        Compute the marginal distribution of this Variable.
        It is assumed that message passing has completed when this function is called.
        Args:
            Z: an optional normalization constant can be passed in. If None is passed, Z is computed.
        Returns: marginal, Z. The first is a numpy array containing the normalized marginal distribution.
         Z is either equal to the input Z, or computed in this function (if Z=None was passed).
        """
        if Z is None:
            Z = None # TODO
        return reduce(reduce_func, self._in_msgs_except()), Z
    
    def send_sp_msg(self, other):
        return self.send_generic_msg(np.multiply, lambda x:x, other)

    def send_ms_msg(self, other):
        return self.send_generic_msg(np.add, np.log, other)

    def send_generic_msg(self, reduce_func, factor_func, other):
        #print bcolors.OKBLUE+"msg ",str(self),"-->",str(other)+bcolors.ENDC
        assert len(self.in_msgs) >= len(self.neighbours) - 1
        assert other in self.neighbours
        assert self.can_send_to(other)

        msgs = self._in_msgs_except(other)

        # element-wise multiplication of all incoming messages
        # (which have the same size, since they operate on the same variable)
        if len(msgs) == 0:
            msg = factor_func(np.array([1.0] * self.num_states))
        else:
            msg = reduce(reduce_func,msgs)
        
        # normalize message to sum=1
        msg_normalized = normalize(msg)

        # put message in destination variable
        other.receive_msg(self,msg_normalized)
        
        if other in self.pending:
            self.pending.remove(other)
    
    def argmax(self,algo):
        if algo == 'sp':
            marginal_func = self.marginal
        elif algo == 'ms':
            marginal_func = self.marginal_ms
        else:
            raise Exception("invalid algorithm")

        return my_argmax(marginal_func()[0])

    def __str__(self):
        # This is printed when using 'print node_instance'
        return "<Variable "+self.name+">"

class Factor(Node):
    def __init__(self, name, f, neighbours):
        """
        Factor node constructor.
        Args:
            name: a name string for this node. Used for printing
            f: a numpy.ndarray with N axes, where N is the number of neighbours.
               That is, the axes of f correspond to variables, and the index along that axes corresponds to a value of that variable.
               Each axis of the array should have as many entries as the corresponding neighbour variable has states.
            neighbours: a list of neighbouring Variables. Bi-directional connections are created.
        """
        # Call the base-class constructor
        super(Factor, self).__init__(name)

        assert len(neighbours) == f.ndim, 'Factor function f should accept as many arguments as this Factor node has neighbours'
        
        for nb_ind in range(len(neighbours)):
            nb = neighbours[nb_ind]
            assert f.shape[nb_ind] == nb.num_states, 'The range of the factor function f is invalid for input %i %s' % (nb_ind, nb.name)
            self.add_neighbour(nb)
            nb.add_neighbour(self)

        self.f = f

    def initialize_messages(self, uniform_value):
        for neigh in self.neighbours:
            self.in_msgs[neigh] = np.array([uniform_value] * neigh.num_states)

    def has_latents(self):
        return any([ neigh.is_latent for neigh in self.neighbours])

    def f_with_observations(self):
        """
        creating an array of coefficients that has the same shape of self.f
        This should be ultimately multiplied by self.f.
        Each observed_state from the neighbours is considered,
        and it's dimensions augmented to its position in the factor space.
        Then, it's tiled in all other dimensions to get a shape as self.f.
        Each of these temporary multi-dimensional arrays are then multiplied 
        element-wise with each other in order to create the final 
        coefficients for self.f.
        For example, if all variables are unobserved the coefficients should result
        in a multi-dimensional array with just ones.
        The more the observed variables, the more the ~0 elements are present
        in the coefficient array.
        """

        # gather all observed_state vectors from the neighbours
        obs_list = [ neigh.observed_state.copy() for neigh in self.neighbours ]
        
        # expand dimension of every vector
        for d in range(len(self.neighbours)):
            for i, obs in enumerate(obs_list):
                if d == i:
                    # expand dimensions only in dimensions that are not the
                    # current variable's one
                    continue
                
                obs = np.expand_dims(obs,d)
                obs_list[i] = obs
        
        # tile the observation vectors to get coefficient matrices
        tile_config_template = [neigh.num_states for neigh in self.neighbours]
        tile_configs = []
        obs_ms = [] # coefficient matrices, one for each neighbour
        for i, obs in enumerate(obs_list):
            tile_config = tile_config_template[:] # copy
            tile_config[i] = 1 # don't tile in own dimension
            obs_m = np.tile(obs,tile_config)
            obs_ms.append(obs_m)
        
        # multiply all temporary coefficient arrays
        final_coefficients = reduce(np.multiply, obs_ms)
        #print "final_coeffients",final_coefficients
        # create the self.f substitute that takes into account observations
        new_f = np.multiply(self.f, final_coefficients)
        return new_f

    def send_sp_msg(self, other):
        return self.send_generic_msg(np.sum, np.multiply, lambda f:f, other)

    def send_ms_msg(self, other):
        return self.send_generic_msg(np.max, np.add, np.log, other)
        
    def send_generic_msg(self, collapse_func, reduce_func, factor_func, other):
        #print bcolors.OKBLUE+"msg ",str(self),"-->",str(other)+bcolors.ENDC
        assert len(self.in_msgs) >= len(self.neighbours) - 1
        assert other in self.neighbours
        assert self.can_send_to(other)

        # outer product/sum of all the messages
        pr = self._in_msgs_reduce(reduce_func, factor_func, other)

        # tiling the product/sum on the axis of the destination message
        other_index = self.neighbours.index(other)
        tile_reps = [1] * len(self.neighbours)
        tile_reps[other_index] = other.num_states
        pr_expanded = np.expand_dims(pr,other_index)
        pr_rep = np.tile(pr_expanded,tuple(tile_reps))
        
        # element-wise product/sum of product of message array/function
        # and factor array/function
        f_o = factor_func(self.f_with_observations())
        summand = reduce_func(pr_rep,f_o)

        # summation/max over all the axes except the destination's one
        msg = self._collapse(collapse_func, summand, other_index)

        # normalize message to sum=1
        msg_normalized = normalize(msg)

        # put message in destination variable
        other.receive_msg(self,msg_normalized)
        
        if other in self.pending:
            self.pending.remove(other)
    
    def marginal_sp(self):
        return self.marginal_generic(np.multiply, lambda f:f)

    def marginal_ms(self):
        return self.marginal_generic(np.add, np.log)

    def marginal_generic(self, reduce_func, factor_func):
        # outer product/sum of all the messages
        pr = self._in_msgs_reduce(reduce_func, factor_func)
        f_o = factor_func(self.f_with_observations())
        ret = reduce_func(pr,f_o)
        return ret

    def __str__(self):
        # This is printed when using 'print node_instance'
        return "<Factor "+self.name+">"
