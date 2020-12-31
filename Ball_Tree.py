import numpy as np
from scipy.spatial import distance

class Ball_Tree():
    """
    This class implements a ball tree with a function for calculating local
    density at a point. It is not meant to be mutable, and has no easy method to
    add nodes once it has been initialized. It maintains indicies into the array
    used to create it at each node, but holds the raw data as well for fast 
    access, but this increases memory usage as you must copy the data.
    Params:
        np.array X: the data array to construct a ball tree of
        int min_pts: the minimum number of points to put in a node before using 
          brute force search
    Attributes:
        np.array self.X: pointer to data
        int min_pts: passed min_pts
        Node root: the root node
    Methods:
        int calc_local_density(pt, epsilon): gets local density at pt
    """

    class Node():
        """
        Used as the primary component of the Ball Tree.
        X is used to denote the original data array
        Params:
            parent,data_pt_i,data_pt,radius,n_elements,data_i=None,data=None
            Same description as in attributes because they are immediately
            made attributes in init
            data and data_i are automatically initialized to None, and should b
            passed only if it is a leaf
        Attributes:
            Node Parent: the parent of this node
            Node left: the left child
            Node right: the right child
            int data_pt_i: index into X of center point of this node
            np.array data_pt: the actual data point
            float radius: the node's radius (max dist from center to other pt)
            int n_elements: num of elements in this branch. includes center
            *Following used only if this is a leaf node. Else None
            np.array data: holds the data in this node including center pt
            np.array data_i: absolute indices into X
        """

        def __init__(self, parent, data_pt_i, data_pt, radius, n_elements, data_i=None, data=None):
            self.parent=parent
            self.right=None
            self.left=None
            #payload
            self.data_pt_i = data_pt_i #the point index into the data
            self.data_pt = data_pt #the actual point
            self.radius = radius#the radius of points nearby
            self.n_elements = n_elements#including this point
            self.data = data #these fields are used for leaf nodes
            self.data_i = data_i 

    def __init__(self, X, min_pts):
        self.X = X
        self.min_pts = min_pts
        centroid = self.X.mean(axis=0)
        center_pt_i = distance.cdist(centroid.reshape(1,-1), self.X).argmin()
        radius = distance.cdist(X[center_pt_i].reshape(1,-1), self.X).max()
        self.root = self._create_node(None, center_pt_i, radius, np.arange(X.shape[0]))
        
    def _find_new_nodes(self, data_subset_i):
        #finds new centers for a node split
        #Params:
            #np.array data_subset_i: indices into the X
        #returns:
            #tuple<int, int> the two node pt indices into X

        centroid = self.X[data_subset_i].mean(axis=0)
        node1_i = data_subset_i[distance.cdist(centroid.reshape(1,-1), self.X[data_subset_i]).argmax()]
        node2_i = data_subset_i[distance.cdist(self.X[node1_i].reshape(1,-1), self.X[data_subset_i]).argmax()]
        return node1_i, node2_i
    
    def _split_data(self, node1_i, node2_i, data_subset_i):
        #splits the data among two node pts, and gives their radii
        #Params:
            #int node1_i: the index into X of the first node
            #int node2_i: the index into X of the second node
            #np.array data_subset_i: indices into the X
        #returns:
            #tuple<np.array, np.array, rad1, rad2>:
                #the indices into X to go with the first node
                #the indices into X to go with the second node
                #the radius of the first node
                #the radius of the second node

        #first col node1, then node2
        distance_arr = distance.cdist(self.X[data_subset_i], self.X[[node1_i, node2_i]])
        cluster_assignments = distance_arr.argmin(axis=1)
        #gives 0 if closer to node_1, 1 if closer to node_2
        data1_local_i = np.nonzero(cluster_assignments == 0)
        data2_local_i = np.nonzero(cluster_assignments == 1)
        cl1 = data_subset_i[data1_local_i]
        cl2 = data_subset_i[data2_local_i]
        rad1 = distance_arr[data1_local_i,0].max()
        rad2 = distance_arr[data1_local_i,1].max()
        return cl1, cl2, rad1, rad2
    
    def _create_node(self, parent_node, data_pt_i, radius, data_subset_i):
        #recursive method to cueate the create a new node
        #(inizializes the tree)
        #Params:
            #Node parent_node: the parent of the node to be constructed
            #np.array data_pt_i: index into X of node to be constructed
            #float radius: radius of node to be constructed
            #np.array data_subset_i: indices into X of node to be constructed
        #returns:
            #the constructed node with left and right children assigned
        if len(data_subset_i) <= self.min_pts:
            #Base Case
            return self.Node(parent_node, data_pt_i, self.X[data_pt_i],radius, data_subset_i.shape[0], 
                        data_subset_i, self.X[data_subset_i])
        else:
            #Create More Nodes
            data_pt1_i, data_pt2_i = self._find_new_nodes(data_subset_i)
            data_subset1_i, data_subset2_i, rad1, rad2 = self._split_data(data_pt1_i, data_pt2_i, data_subset_i)
            current_node = self.Node(parent_node,data_pt_i, self.X[data_pt_i], radius, len(data_subset_i))
            current_node.right = self._create_node(current_node, data_pt1_i, rad1, data_subset1_i)
            current_node.left = self._create_node(current_node, data_pt2_i, rad2, data_subset2_i)
            return current_node
        
    def calc_local_density(self, pt, epsilon):
        #provides nice user interface
        #recursion is used for elegance, but if this is applied to big data, iteration should be implemented
        #do not call with node specified. This param is for recursive calls
        #params: incomplete
            #np.array data_pt: shape (n) the point at which to calculate the density
            #float epsilon: the radius to consider density
        return int(self._calc_local_density(pt, epsilon, self.root)) #casts from np.int64
    
    def _calc_local_density(self, pt, epsilon, node):
        dist = distance.euclidean(pt, node.data_pt)
        if dist - node.radius > epsilon:
            #base1: nothing along below this node is within epsilon
            return 0 
        elif dist + node.radius <= epsilon:
            #base2: everything in this node is within epsilon
            return node.n_elements
        elif node.data is not None:
            #base3: we've reached a leaf
            return (distance.cdist(pt.reshape(1,-1),node.data)[0] <= epsilon).sum()
        else:
            #recursive case
            return self._calc_local_density(pt, epsilon, node.right) + self._calc_local_density(pt, epsilon, node.left)
