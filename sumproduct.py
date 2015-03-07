#!/usr/bin/env python
import numpy as np

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
        print "ERROR: why sum(vector)==0???",vector
        return vector

    return vector/s

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
   
    def send_sp_pending(self):
        for node in list(self.pending):
            print "in",self,"link to",node,"is pending. sending message..."
            self.send_sp_msg(node)

    def receive_msg(self, other, msg):  
        print self.name,"received message",msg,"from",str(other)
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

    def _in_msgs_product(self,exclude=None):
        msg_vectors = self._in_msgs_except(exclude)

        if len(msg_vectors) == 0:
            # leaf factor?
            return np.array(1)

        ixstuff = np.ix_(*msg_vectors)
        ret = reduce(np.multiply, ixstuff)
        return ret

    def _sum_collapse(self, summand, exclude=None):
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
            ret = np.sum(ret,axis=i)
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
        
    def reset(self):
        super(Variable, self).reset()
        self.observed_state = np.ones(self.num_states)
        
    def marginal(self, Z=None):
        """
        Compute the marginal distribution of this Variable.
        It is assumed that message passing has completed when this function is called.
        Args:
            Z: an optional normalization constant can be passed in. If None is passed, Z is computed.
        Returns: marginal, Z. The first is a numpy array containing the normalized marginal distribution.
         Z is either equal to the input Z, or computed in this function (if Z=None was passed).
        """
        # TODO: compute marginal
        return None, Z
    
    def send_sp_msg(self, other):
        print bcolors.OKBLUE+"msg ",str(self),"-->",str(other)+bcolors.ENDC
        assert len(self.in_msgs) >= len(self.neighbours) - 1
        assert other in self.neighbours
        assert self.can_send_to(other)

        msgs = self._in_msgs_except(other)

        # element-wise multiplication of all incoming messages
        # (which have the same size, since they operate on the same variable)
        if len(msgs) == 0:
            msg = np.array([1] * self.num_states)
        else:
            msg = reduce(np.multiply,msgs)
        
        # normalize message to sum=1
        msg_normalized = normalize(msg)

        # put message in destination variable
        other.receive_msg(self,msg_normalized)
        
        if other in self.pending:
            self.pending.remove(other)

    def send_ms_msg(self, other):
        # TODO: implement Variable -> Factor message for max-sum
        pass
    
    def marginal(self):
        return reduce(np.multiply,self._in_msgs_except())

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
        print "final_coeffients",final_coefficients
        # create the self.f substitute that takes into account observations
        new_f = np.multiply(self.f, final_coefficients)
        return new_f

    def send_sp_msg(self, other):
        print bcolors.OKBLUE+"msg ",str(self),"-->",str(other)+bcolors.ENDC
        assert len(self.in_msgs) >= len(self.neighbours) - 1
        assert other in self.neighbours
        assert self.can_send_to(other)

        # outer product of all the messages
        pr = self._in_msgs_product(other)

        # tiling the product on the axis of the destination message
        other_index = self.neighbours.index(other)
        tile_reps = [1] * len(self.neighbours)
        tile_reps[other_index] = other.num_states
        pr_expanded = np.expand_dims(pr,other_index)
        pr_rep = np.tile(pr_expanded,tuple(tile_reps))
        
        # element-wise product of product of message array/function
        # and factor array/function
        f_o = self.f_with_observations()
        summand = np.multiply(pr_rep,f_o)

        # summation over all the axes except the destination's one
        msg = self._sum_collapse(summand, other_index)

        # normalize message to sum=1
        msg_normalized = normalize(msg)

        # put message in destination variable
        other.receive_msg(self,msg_normalized)
        
        if other in self.pending:
            self.pending.remove(other)
   
    def send_ms_msg(self, other):
        # TODO: implement Factor -> Variable message for max-sum
        pass

    def __str__(self):
        # This is printed when using 'print node_instance'
        return "<Factor "+self.name+">"
