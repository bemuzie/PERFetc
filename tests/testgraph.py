__author__ = 'denest'
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import sys


class Compartment(nx.DiGraph):

    def add_node(self, n, attr_dict=None, **attr):
        """Add a single node n and update node attributes.

        Parameters
        ----------
        n : node
            A node can be any hashable Python object except None.
        attr_dict : dictionary, optional (default= no attributes)
            Dictionary of node attributes.  Key/value pairs will
            update existing data associated with the node.
        attr : keyword arguments, optional
            Set or change attributes using key=value.

        See Also
        --------
        add_nodes_from

        Examples
        --------
        >>> G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_node(1)
        >>> G.add_node('Hello')
        >>> K3 = nx.Graph([(0,1),(1,2),(2,0)])
        >>> G.add_node(K3)
        >>> G.number_of_nodes()
        3

        Use keywords set/change node attributes:

        >>> G.add_node(1,size=10)
        >>> G.add_node(3,weight=0.4,UTM=('13S',382871,3972649))

        Notes
        -----
        A hashable object is one that can be used as a key in a Python
        dictionary. This includes strings, numbers, tuples of strings
        and numbers, etc.

        On many platforms hashable items also include mutables such as
        NetworkX Graphs, though one should be careful that the hash
        doesn't change on mutables.
        """
        # set up attribute dict
        if attr_dict is None:
            attr_dict=attr
        else:
            try:
                attr_dict.update(attr)
            except AttributeError:
                raise NetworkXError(\
                    "The attr_dict argument must be a dictionary.")
        if n not in self.succ:
            self.succ[n] = {}
            self.pred[n] = {}
            self.node[n] = {'distribution':'norm',
                            'distpars':[20,3],
                            'conc':np.zeros(self.graph['time'].size)
                            }
            self.node[n].update(attr_dict)
        else: # update attr even if node already exists
            self.node[n].update(attr_dict)
        self.node[n]['outf']= getattr(stats, self.node[n]['distribution']).pdf(self.graph['time'], *self.node[n]['distpars'])
        self.node[n]['outf']/=np.sum(self.node[n]['outf'])
    def add_nodes_from(self, nodes, **attr):
        """Add multiple nodes.

        Parameters
        ----------
        nodes : iterable container
            A container of nodes (list, dict, set, etc.).
            OR
            A container of (node, attribute dict) tuples.
            Node attributes are updated using the attribute dict.
        attr : keyword arguments, optional (default= no attributes)
            Update attributes for all nodes in nodes.
            Node attributes specified in nodes as a tuple
            take precedence over attributes specified generally.

        See Also
        --------
        add_node

        Examples
        --------
        >>> G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_nodes_from('Hello')
        >>> K3 = nx.Graph([(0,1),(1,2),(2,0)])
        >>> G.add_nodes_from(K3)
        >>> sorted(G.nodes(),key=str)
        [0, 1, 2, 'H', 'e', 'l', 'o']

        Use keywords to update specific node attributes for every node.

        >>> G.add_nodes_from([1,2], size=10)
        >>> G.add_nodes_from([3,4], weight=0.4)

        Use (node, attrdict) tuples to update attributes for specific
        nodes.

        >>> G.add_nodes_from([(1,dict(size=11)), (2,{'color':'blue'})])
        >>> G.node[1]['size']
        11
        >>> H = nx.Graph()
        >>> H.add_nodes_from(G.nodes(data=True))
        >>> H.node[1]['size']
        11

        """
        for n in nodes:
            self.add_node(n, **attr)

    def add_edge(self, u, v, attr_dict=None, **attr):
        """Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Edge attributes can be specified with keywords or by providing
        a dictionary with key/value pairs.  See examples below.

        Parameters
        ----------
        u,v : nodes
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.
        attr_dict : dictionary, optional (default= no attributes)
            Dictionary of edge attributes.  Key/value pairs will
            update existing data associated with the edge.
        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        See Also
        --------
        add_edges_from : add a collection of edges

        Notes
        -----
        Adding an edge that already exists updates the edge data.

        Many NetworkX algorithms designed for weighted graphs use as
        the edge weight a numerical value assigned to a keyword
        which by default is 'weight'.

        Examples
        --------
        The following all add the edge e=(1,2) to graph G:

        >>> G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> e = (1,2)
        >>> G.add_edge(1, 2)           # explicit two-node form
        >>> G.add_edge(*e)             # single edge as tuple of two nodes
        >>> G.add_edges_from( [(1,2)] ) # add edges from iterable container

        Associate data to edges using keywords:

        >>> G.add_edge(1, 2, weight=3)
        >>> G.add_edge(1, 3, weight=7, capacity=15, length=342.7)
        """
        # set up attribute dict
        if attr_dict is None:
            attr_dict=attr
        else:
            try:
                attr_dict.update(attr)
            except AttributeError:
                raise NetworkXError(\
                    "The attr_dict argument must be a dictionary.")
            # add nodes
        if u not in self.succ:
            self.succ[u]={}
            self.pred[u]={}
            self.add_node(u)
        if v not in self.succ:
            self.succ[v]={}
            self.pred[v]={}
            self.add_node(v)
            # add the edge
        datadict=self.adj[u].get(v,{})
        datadict.update(attr_dict)
        self.succ[u][v]=datadict
        self.pred[v][u]=datadict

    def add_edges_from(self, ebunch, attr_dict=None, **attr):
        """Add all the edges in ebunch.

        Parameters
        ----------
        ebunch : container of edges
            Each edge given in the container will be added to the
            graph. The edges must be given as as 2-tuples (u,v) or
            3-tuples (u,v,d) where d is a dictionary containing edge
            data.
        attr_dict : dictionary, optional (default= no attributes)
            Dictionary of edge attributes.  Key/value pairs will
            update existing data associated with each edge.
        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.


        See Also
        --------
        add_edge : add a single edge
        add_weighted_edges_from : convenient way to add weighted edges

        Notes
        -----
        Adding the same edge twice has no effect but any edge data
        will be updated when each duplicate edge is added.

        Examples
        --------
        >>> G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_edges_from([(0,1),(1,2)]) # using a list of edge tuples
        >>> e = zip(range(0,3),range(1,4))
        >>> G.add_edges_from(e) # Add the path graph 0-1-2-3

        Associate data to edges

        >>> G.add_edges_from([(1,2),(2,3)], weight=3)
        >>> G.add_edges_from([(3,4),(1,4)], label='WN2898')
        """
        # set up attribute dict
        if attr_dict is None:
            attr_dict=attr
        else:
            try:
                attr_dict.update(attr)
            except AttributeError:
                raise NetworkXError(\
                    "The attr_dict argument must be a dict.")
            # process ebunch
        for e in ebunch:
            ne = len(e)
            if ne==3:
                u,v,dd = e
                assert hasattr(dd,"update")
            elif ne==2:
                u,v = e
                dd = {}
            else:
                raise NetworkXError(\
                    "Edge tuple %s must be a 2-tuple or 3-tuple."%(e,))
            if u not in self.succ:
                self.succ[u] = {}
                self.pred[u] = {}
                self.add_node(u)
            if v not in self.succ:
                self.succ[v] = {}
                self.pred[v] = {}
                self.add_node(v)
            datadict=self.adj[u].get(v,{})
            datadict.update(attr_dict)
            datadict.update(dd)
            self.succ[u][v] = datadict
            self.pred[v][u] = datadict

    def concentration(self,node):
        thisnode=self.node[node]
        if self.predecessors(node)==[]:
            pass
        else:
            inputconc=np.sum( [self.node[inputs]['conc'] * self.edge[inputs][node]['weight']
                              for inputs in self.predecessors_iter(node)],axis=0)
            concnew=np.convolve( thisnode['outf'], inputconc)[:self.graph['time'].size]
            diff=thisnode['conc']-concnew
            thisnode['conc']=concnew
            if np.sum(diff*diff)>0.000000000001:
                for i in self.successors_iter(node):
                    self.concentration(i)




sys.setrecursionlimit(50000)

CVS=Compartment()
CVS.graph['time']=np.arange(0,200,.1)
CVS.graph['i']=0
allnodes=["LV","RV",'Head','Lung','Liver','Legs','Kidney','Injection']


CVS.add_nodes_from(allnodes)
i=0
for n in CVS:
    CVS.node[n]['color']=(np.random.rand(),np.random.rand(),np.random.rand())

CVS.node['Injection']['conc'][1:100]=200


#CVS.add_path(['LV','Kidney','RV'])
CVS.add_path(['LV','Head','RV'])
#CVS.add_path(['LV','Liver','RV'])
CVS.add_path(['LV','Legs','RV'])
CVS.add_path(['RV','Lung','LV'])


for i in [('RV','Lung'),
            ('Lung','LV')]:
    CVS.add_weighted_edges_from([i+(1,)])

CVS.add_weighted_edges_from([('Injection','RV',0.1)])
CVS.add_weighted_edges_from([('Head','RV',0.2)])
CVS.add_weighted_edges_from([('Legs','RV',0.5)])
CVS.add_weighted_edges_from([('Liver','RV',0.3)])
CVS.add_weighted_edges_from([('LV','Head',1)])
CVS.add_weighted_edges_from([('LV','Legs',1)])
CVS.add_weighted_edges_from([('LV','Liver',1)])

CVS.add_node('RV',{'distpars':[3,1]})
CVS.add_node('LV',{'distpars':[6,1]})
CVS.add_node('Lung',{'distribution':'gamma','distpars':[6,3,2]})
CVS.add_node('Legs',{'distpars':[30,9]})
CVS.add_node('Head',{'distpars':[20,4]})
CVS.add_node('Liver',{'distribution':'gamma','distpars':[1.4,0,10]})


nx.write_gpickle(CVS, '../graphs/newgraph.pickle')

CVS2=nx.read_gpickle('../graphs/newgraph.pickle')
#nx.draw(CVS2)

CVS2.concentration('RV')
plt.plot(CVS2.graph['time'],CVS2.node['RV']['conc'],label='RV')
plt.plot(CVS2.graph['time'],CVS2.node['Lung']['conc'],label='Lung')
plt.plot(CVS2.graph['time'],CVS2.node['LV']['conc'],label='LV')
plt.plot(CVS2.graph['time'],CVS2.node['Legs']['conc'],label='Legs')
plt.plot(CVS2.graph['time'],CVS2.node['Head']['conc'],label='Head')
plt.plot(CVS2.graph['time'],CVS2.node['Liver']['conc'],label='Liver')

plt.legend()
plt.show()
