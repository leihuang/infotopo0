"""
- Add LaTeX capacity? (dot2tex?)

When one makes a graph, he shouldn't need to heed to the global picture. 
He should just include all the nodes from geodesic explorations. 
And after each level, some codes are run to consolidate the nodes and edges. 

Can be used to encode an MBAM path...

"""

from __future__ import division
from collections import OrderedDict as OD, Mapping
import itertools
import copy
import cPickle
import re
import random

import networkx as nx
import numpy as np
        
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from util import butil, plotutil
reload(plotutil)


class HasseDiagram(nx.DiGraph):
    """
    Graded...
    
    https://en.wikipedia.org/wiki/Graded_poset
    
    rank of a node:
    rank of a diagram: 
    corank:
    rank level: 
    
    Default nodeid naming convention: a1, a2, b1, b2, b3, c1, ...
    
    Subclassing networkx.DiGraph. 
    
    nodes and edges: forget about them; wrote my own
     |
     V
    Some convenient methods of the class:
        nx.DiGraph.nodes(self, data=False): return a list of nodeids; 
            if data=True, return a dictionary mapping from nodeid to node attrs
        nx.DiGraph.edges(self, nbunch=None, data=False): return a list of edges
            of the provided nodeids (if given), with edge attrs (if data=True)
        nx.DiGraph.successors
        nx.DiGraph.predecessors
        
    
    """
    
    def __init__(self, rank=None, hd=None, **kwargs):  # __init__(self, rank, nodes, edges, network):
        """
        Should it specify the rank (depth) at the beginning?
        
        Custom attributes: 
            rank:
            order:
            quotient:
            
        Special care is needed for custom attributes when creating new instances
        from existing ones, as they are prone to be lost. 
            
        
        Another utility potentially to be included: construct a HasseDiagram
        instance from a butil.DF, like that returned by self.nodeattrs
        
        order: networkx do not keep track of the order of nodes added; but for 
            a Hasse diagram, the order of nodes (of the same rank) matters
        
        Input:
            rank:
            hd: an attribute dict (from eg, hd.__dict__) or 
                a HasseDiagram instance  
        """
        super(HasseDiagram, self).__init__(**kwargs)
        if rank is not None:
            self.rank = rank
            self.order = OD(zip(range(rank,-2,-1), [[] for _ in range(rank+1)])) 

            # Shall I call 'symmetry' 'eqcls' instead?
            # Presumably not all equivalent classes are due to symmetry (?) 
            self.symmetry = OD(zip(range(rank,-2,-1), [[] for _ in range(rank+1)])) 
            
            self.eqclss = OD(zip(range(rank,-2,-1), [[] for _ in range(rank+1)]))
            
            
        if hd is not None:
            if hasattr(hd, '__dict__'):
                attrs = hd.__dict__
            else:
                attrs = hd
            for k, v in attrs.items():
                setattr(self, k, v)
                
    __init__.__doc__ = nx.DiGraph.__init__.__doc__
        
    
    def __repr__(self):
        return self.nodeattrs.__repr__()
    
    """
    This code makes the whole thing brittle, and the return isn't that great 
    (only for convenience). In general, this type of coding style should be 
    avoided in the prototyping phase, as it can easily create bugs and suck up
    much time.  
    
    def __getattr__(self, attrid):
        if attrid in self.nodes():
            return self.get_nodeattr(attrid, to_ser=True)
        else:
            
            raise AttributeError()
    """

            
    def add_node(self, nodeid, corank=None, rank=None, style='filled', **kwargs):
        """
        Essential attributes of a node:
            nodeid: a str, preferably short, for internal manipulation
            corank: an int, more convenient than providing rank
            
            # label: for drawing; if not given, default to nodeid -- decide to 
                remove the attr to further lean the codes; defer the attr
                setting till drawing; because the utility of a hasse diagram
                is not always for visualization, but also model composition...
            
            kwargs: hold a bunch of objects and attributes; examples of 
                objects: ratelaw.RateLaw, predict.Predict, residual.Fit
            
        """
        if corank is not None:
            rank = self.rank - corank
        if rank is not None:
            corank = self.rank - rank
        super(HasseDiagram, self).add_node(nodeid, corank=corank, rank=rank, 
                                           style=style, **kwargs)
        self.order[rank].append(nodeid)
        
    
    def add_edge(self, nodeid1, nodeid2, edgeid=None, style='solid', **kwargs):
        """
        Input:
            style: for drawing
        """
        assert self.get_nodeattr(nodeid1)['rank'] ==\
            self.get_nodeattr(nodeid2)['rank'] + 1 
        if edgeid is None:
            edgeid = (nodeid1, nodeid2)
        super(HasseDiagram, self).add_edge(nodeid1, nodeid2, edgeid=edgeid,
                                           style=style, **kwargs)
    
    
    def rm_node(self, nodeid):
        self.remove_node(nodeid)
        for rank in self.order.keys():
            if nodeid in self.order[rank]:
                self.order[rank].remove(nodeid) 
        
    @property
    def nodeids(self):
        """Return an ordered list of node ids.
        """
        return butil.flatten(self.order.values(), depth=1)  # order-preserving
    
    
    @property
    def nodeattrs(self):
        """Return a DataFrame view of all node attributes.
        """
        nodeid2attr = dict(self.nodes(data=True))
        nodeattrs = butil.get_values(nodeid2attr, self.nodeids) 
        return butil.DF(nodeattrs, index=self.nodeids)
    
    
    @property
    def edgeids(self):
        """Return an ordered list of edge ids.
        """
        edgeids_all = []  # a sorted list of the edges in a "complete" Hasse diagram
        for nodeids1, nodeids2 in zip(self.order.values()[:-1], self.order.values()[1:]):
            edgeids_all.extend(butil.get_product(nodeids1, nodeids2))
        edgeids_unordered = self.edges() 
        edgeids_ordered = [edgeid for edgeid in edgeids_all 
                           if edgeid in edgeids_unordered]
        return edgeids_ordered
    
    
    @property
    def edgeattrs(self):
        """
        """
        edgedata = self.edges(data=1)
        df = butil.DF(dict([((row[0],row[1]), row[2]) for row in edgedata])).T
        df.index = df.index.tolist()  # change the default multiindex to index
        return df.reindex(self.edgeids)  # reordering
    
    
    def get_nodeattr(self, nodeid, to_ser=False):
        """Return either a Series view or a dict of the attrs of a node 
        for modification.
        
        Input:
            to_ser: default to be False and return a dict for **setting attrs**
                (eg, hd.get_nodeattr('a1')['size'] = 1); if True, return a 
                butil.Series for the ease of visualization
        """
        nodeattr = [item[1] for item in self.nodes(data=True) 
                    if item[0]==nodeid][0]  # a dict
        if to_ser:
            return butil.Series(nodeattr)
        else:
            return nodeattr
    get_node = get_nodeattr  # deprecation warning
    
    
    def get_nodeids_rank(self, rank=None, corank=None):
        """Get a list of nodeids with the given rank, aka "rank level".
        
        Deprecation warning... 
        """
        print "'get_nodeids_rank': deprecated"
        return self.order[rank]
    
    
    @property
    def feqorder(self):
        """flat eq-order."""
        assert hasattr(self, 'eqorder')
        feqorder = OD.fromkeys(self.eqorder.keys(), None)
        for rank, nodeids_rank in self.eqorder.items():
            nodeids_rank = butil.flatten([[_] if not isinstance(_, list)
                                          else _ for _ in nodeids_rank], 1)
            feqorder[rank] = nodeids_rank
        return feqorder
    
    
    def get_pos(self, width=1, height=1, stack=False, offset=0.03):
        """
        """
        pos = OD()
        """
        for rank in range(self.rank+1):
            nodeids_rank = self.get_nodeids_rank(rank=rank)
            print nodeids_rank
            for idx, nodeid in enumerate(nodeids_rank):
                y = rank / self.rank
                if rank == self.rank:
                    x = 0.5
                else:
                    x = (idx+1) / (len(nodeids_rank)+1)
                pos[nodeid] = (x, y)
        return pos
    
        """
        
        if hasattr(self, 'eqorder'):
            # use eqorder
            if stack:
                eqorder = self.eqorder
            else:
                eqorder = self.feqorder  # flattened
            for rank, nodeids_rank in eqorder.items():
                doubleids_rank = []  # doubleton nodeids
                otherids_rank = []  # other nodeids (singletons, tripletons, etc.)
                for nodeid in nodeids_rank:
                    if isinstance(nodeid, list):
                        otherids_rank.extend(nodeid)
                        doubleids_rank.append(())
                    else:
                        doubleids_rank.append(nodeid)
                        
                n = len(otherids_rank)
                offsetidxs = np.linspace(-(n-1)/2, (n-1)/2, n)
                for idx, nodeid in enumerate(otherids_rank):
                    x = 0.5 * width
                    y = ((rank+1) / (self.rank+2) + offsetidxs[idx]*offset) * height
                    pos[nodeid] = (x, y)
                    
                for idx, nodeid in enumerate(doubleids_rank):
                    if nodeid:
                        x = (idx+1) / (len(doubleids_rank)+1) * width
                        y = (rank+1) / (self.rank+2) * height
                        pos[nodeid] = (x, y)        
                    
        else:
            # use order
            for rank, nodeids_rank in self.order.items():  
                for idx, nodeid in enumerate(nodeids_rank):
                    x = (idx+1) / (len(nodeids_rank)+1) * width
                    y = (rank+1) / (self.rank+2) * height
                    pos[nodeid] = (x, y)
        return pos
        
    
    """
    def plot(self, node2label, height=100, width=100, crop=True, border=3,
             arrowsize=0.5, nodeshape='box', labelfontsize=8,  
             colornamemap='white', scalemap=0.2, filepath=''):
        
        Need to add:
            - Fix the box sizes for each level (level2hw)
            - Manually set the edge pos through the box sizes and node pos
        
        Input:
            nodeshape: 'box', 'circle', 'ellipse', etc. 
                (http://www.graphviz.org/doc/info/shapes.html)
            colornamemap: a mapping or a str;
                (http://www.graphviz.org/content/color-names; note that
                 while graphviz supports all the colorname schemes mentioned
                 in the link, pygraphviz seems to support only the X11 scheme.)
            
        
        import pygraphviz as pgv
        import dot2tex
        
        g = pgv.AGraph(strict=False, directed=True, splines='polyline', 
                       overlap='scale')  # splines='line','false','polyline' all don't seem to work
        g.add_nodes_from(self.nodeids)
        g.add_edges_from(self.edges(), arrowsize=arrowsize)
        
        if isinstance(colornamemap, str):
            colornamemap = dict.fromkeys(self.nodeids, colornamemap)
        if isinstance(scalemap, float):
            scalemap = dict.fromkeys(self.nodeids, scalemap)
        
        pos = self.get_pos(height=height, width=width)
        for nodeid, nodepos in pos.items():
            node = g.get_node(nodeid)
            # graphviz object attributes reference: 
            # http://www.graphviz.org/content/attrs
            node.attr['pos'] = '%f, %f'%tuple(nodepos)
            node.attr['shape'] = nodeshape
            node.attr['fillcolor'] = colornamemap[nodeid]
            node.attr['style'] = 'filled'
            node.attr['size'] = 2.
            node.attr['width'] = scalemap[nodeid]
            node.attr['height'] = scalemap[nodeid]
            node.attr['label'] = self.node[nodeid]['label']
            node.attr['fontsize'] = labelfontsize
        
        for idx, nodeid in enumerate(self.nodeids):
            for nodeid2 in self.neighbors(nodeid):
                g.get_edge(nodeid, nodeid2).attr['sametail'] = str(idx)
                g.get_edge(nodeid, nodeid2).attr['weight'] = 100.  # doesn't seem to work
        
        for idx, nodeid in enumerate(self.nodeids):
            for nodeid0 in self.predecessors(nodeid):
                g.get_edge(nodeid0, nodeid).attr['samehead'] = str(idx + self.number_of_nodes())
        
        #import ipdb
        #ipdb.set_trace()
        
        #g.node_attr['texmode'] = 'math'
        #g.layout(prog='neato')
        
        #g.write('.tmp.dot')
        g.write(path='.tmp.xdot')  #, args='-Gsplines=false')
        fhr = open('.tmp.xdot')
        dotcode = ''.join(fhr.readlines())
        fhr.close()
        
        texcode = dot2tex.dot2tex(dotcode, format='tikz', texmode='math', 
                                  crop=crop)
        if crop:
            texcode = texcode.replace('Border{0pt}', 'Border{%dpt}'%border)
        
        fhw = open(filepath, 'w')
        fhw.write(texcode)
        fhw.close()
        
        #g.draw('tmp/tmp2.dot', prog='neato', args='-n2')
        #g.draw('tmp/tmp3.xdot', prog='neato', args='-n2')
        #g.draw(filepath, prog='neato', args='-n2')

    """
    
    def draw(self, nodeid2pos=None, nodeid2label=None, nodeid2color='white',
             edgeid2color='black', stack=False,
             width=1000, height=1000, nodeshape='box',
             rank2size=None, margin=10, filepath=''):
        """Draw the diagram.
        
        FIXME ***: Fix the box sizes... Control the figure size... 
        
        Workflow:
        1. Convert it to a pygraphviz.AGraph instance
        2. Use pygraphviz to save it to an xdot file
        3. Use dot2tex to convert the xdot file to a tex file
        4. Use pdf2latex to compile it to a pdf file
        
        Input:
            nodeid2label: a func or a mapping; if None, default to use nodeids 
                as labels
            nodeid2color: a color (str, rgb tuple, etc.) or a mapping
            rank2size: if give, a mapping specifying the box sizes (if 
                nodeshape=='box'); box sizes are tuples of (width, height);
                note that for some sizes the box boundaries would disappear, 
                and trying slightly different sizes is needed to have the box
                reappear
            nodeshape: 'box', 'circle', 'ellipse', 'point', etc. 
                http://www.graphviz.org/doc/info/shapes.html
            margin:
        """
        import pygraphviz as pgv
        import dot2tex
        import subprocess
        
        ## Get positions, labels and colors of the nodes
        if nodeid2pos is None:
            nodeid2pos = self.get_pos(width=width, height=height,
                                      stack=stack)
        
        # get nodeid2label
        if nodeid2label is None:
            nodeid2label = dict(zip(self.nodeids, self.nodeids))
        elif hasattr(nodeid2label, '__call__'):
            nodeid2label = dict(zip(self.nodeids, map(nodeid2label, self.nodeids)))
        else:
            pass
        # get nodeid2color
        if not isinstance(nodeid2color, dict):
            nodeid2color = dict.fromkeys(self.nodeids, nodeid2color)
        # get nodeid2style
        nodeid2style = self.nodeattrs.style.to_dict()
        # get edgeid2style
        edgeid2style = self.edgeattrs.style.to_dict()
        # get edgeid2color
        if not isinstance(edgeid2color, Mapping):
            edgeid2color = dict.fromkeys(self.edgeids, edgeid2color)
        if rank2size is None:
            rank2size = dict.fromkeys(self.order.keys(), (1,1))

        
        ## Create a pygraphviz.AGraph instance
        ## options for splines: 'line','false','polyline', 'curved', 'ortho', etc. 
        g = pgv.AGraph(strict=False, directed=True, splines='line', 
                       overlap='scale')
        g.add_nodes_from(self.nodeids)
        g.add_edges_from(self.edges())  #, arrowsize=arrowsize)

        #for nodeid1, nodeid2, edgeattr in self.edges(data=True):
        #    edge = g.get_edge(nodeid1, nodeid2)
        #    edge.attr['label'] = edgeattr.get('info', '')
        
        _colormap = lambda s: {'light red': '#FF8080',
                               'light blue': '#8080FF'}.get(s, s)

        for rank, nodeids_rank in self.order.items():
            for nodeid in nodeids_rank:
                node = g.get_node(nodeid)
                # graphviz object attributes reference: 
                # http://www.graphviz.org/content/attrs
                node.attr['pos'] = '%f,%f' % nodeid2pos[nodeid]
                node.attr['label'] = nodeid2label[nodeid]
                node.attr['fillcolor'] = _colormap(nodeid2color.get(nodeid, 'white'))
                node.attr['shape'] = nodeshape
                node.attr['style'] = nodeid2style[nodeid]
                node.attr['fontsize'] = 50
                if rank2size:
                    rw, rh = rank2size[rank]
                    node.attr['width'] = rw #* width
                    node.attr['height'] = rh #* height
                    node.attr['fixedsize'] = True
                
        """
        # need to change the following two blocks as they probably don't work
        # samehead and sametail only work for dot, but I need to use neato (-n)
        # to fix the node positions
        for idx, nodeid in enumerate(self.nodeids):
            for nodeid2 in self.neighbors(nodeid):
                g.get_edge(nodeid, nodeid2).attr['sametail'] = str(idx)
                g.get_edge(nodeid, nodeid2).attr['weight'] = 100.  # doesn't seem to work
        
        for idx, nodeid in enumerate(self.nodeids):
            for nodeid0 in self.predecessors(nodeid):
                g.get_edge(nodeid0, nodeid).attr['samehead'] = str(idx + self.number_of_nodes())
        """
        
        for edge in g.edges():
            edge.attr['tailport'] = 's'  # 'south' and 'north' (I guess)
            edge.attr['headport'] = 'n'
        
        for edgeid in self.edgeids:
            g.get_edge(*edgeid).attr['style'] = edgeid2style[edgeid]
            g.get_edge(*edgeid).attr['color'] = edgeid2color[edgeid]
        
        ## Write the pygraphviz.AGraph instance to an dot file (in xdot format)
        import os
        ready = raw_input('Are you ready to litter this directory with all '\
            'the debris produced by pdflatex, etc.?\n%s  (y or n) '% os.getcwd())
        if ready not in ['y', 'Y', 'yes', 'Yes']:
            raise Exception
        fileid = str(self.nodeids[0]).replace("'","").replace(', ','_').\
            replace('(','').replace(')','')
        filepath_dot = '.%s.dot' % fileid
        filepath_tex = '.%s.tex' % fileid
        filepath_pdf = '.%s.pdf' % fileid
        g.write(path=filepath_dot)
        
        ## Grab the content of the xdot file and convert it to tex file
        ## content using dot2tex
        fh_dot = open(filepath_dot)
        dotcode = ''.join(fh_dot.readlines())
        fh_dot.close()
        
        """
        Problems of different formats: 
            'pgf': often misses box frames, in the process of dot2tex (I 
                confirmed that tex file misses the frames and dot file is fine)
            'tikz': does not fix box sizes or always draws edges as curves;
                update: curved edges are fixed by setting "straightedges=True"
            'pst' (PSTricks): compilation often fails
        
        Note (2016-11-07): Finally tracked down the problem for 'pgf' missing 
        some box frames. In dot2tex.py (/Library/Python/2.7/site-packages/dot2tex/)
        line 263, the functional body of "parse_drawstring", it was using 'int'
        to parse the coordinate strings of polygons (boxes), but int('2.5')
        would throw ValueError, so I changed it to 'float', and no more missing
        boxes! 
        (I was coding a make-shift solution of manually changing the 'texcode' 
        string outputted by dot2tex; I thought about adding the box information 
        to it, and have to infer the size information in the first place, but 
        that does not guarantee to work as some rows can be all missing etc.; 
        I spent about two hours on without success and it was messy. Then I 
        realized that since dot file was fine but tex file was problematic, 
        something was wrong in the conversion; and fortunately the conversion 
        was done in a Python file, https://github.com/kjellmf/dot2tex, 
        so it took me about two more hours to track down the problem.)  
        """   
        texcode = dot2tex.dot2tex(dotcode, texmode='raw', format='pgf',  
                                  crop=True, valignmode='center',  # 'dot' 
                                  prog='neato', progoptions='-n',
                                  straightedges=True, margin='%dpt'%margin,
                                  switchdraworder=False, usepdflatex=True, 
                                  nominsize=True)
        
        """
        def _clean_tex(texcode):
            # add boxes
            # customize sizes
            
            ## for each node:
            # comment out the line
            # parse out the center coord
            # calculate the four corner coords
            # make the line and stick it above
            
            nodeid2coords = OD()
            parts = texcode.split('% Node: ')
            
            def _str2coord(s):
                s = re.sub('[bp\s\(\)]', '', s)
                return np.asarray(s.split(','), dtype='float')
                
            for part in parts[1:]:
                nodeid = part.split('\n')[0]
                #centerstr = re.search('(?<=draw).*(?=node)', part).group()
                #centercoord = _str2coord(centerstr)
                try:
                    cornerstr = re.search('(?<=filldraw).*(?=cycle)', part).group()
                    cornerstrs = cornerstr.split('--')[:-1]                
                    cornercoords = [_str2coord(s) for s in cornerstrs]
                    nodeid2coords[nodeid] = cornercoords
                except:
                    pass
                #nodeid2coords[nodeid] = [centercoord] + cornercoords
            
            def _coords2size(coords):
                xs = [coord[0] for coord in coords]    
                ys = [coord[1] for coord in coords]
                return max(xs) - min(xs), max(ys)- min(ys)
            
            nodeid2size = butil.chvals(nodeid2coords, _coords2size)
            rank2size = [(self.get_nodeattr(nodeid)['rank'], size) 
                         for nodeid, size in nodeid2size.items()]    
                
            #import ipdb
            #ipdb.set_trace()
             
                
            #texcode = texcode.replace('\\filldraw', '%\\filldraw')

            nodeid2line = OD()
            for nodeid, pos in nodeid2pos.items():
                rank = [rank for rank, nodeids_rank in self.order.items()
                        if nodeid in nodeids_rank][0]
                rw, rh = rank2size[rank]
                w, h = rw*width, rh*height
                x, y = pos
                coords = (x+w/2, y+h/2, x+w/2, y-h/2, x-w/2, y-h/2, x-w/2, y+h/2)
                line = '\\filldraw (%fbp,%fbp) -- (%fbp,%fbp) -- (%fbp,%fbp) -- (%fbp,%fbp) -- cycle;'%coords
                nodeid2line[nodeid] = line
            
            parts = texcode.split('% Node: ')
            parts2 = [parts[0]]
            for part in parts[1:]:
                nodeid = part.split('\n')[0]
                part2 = part.replace('\\draw (', nodeid2line[nodeid]+'\n\\draw (')
                parts2.append(part2)
            #texcode = '% Node: '.join(parts2)

            #return texcode
            return rank2size
        
        #texcode = _clean_tex(texcode)
        """
    
        ## Dump the tex file content to a tex file
        fh_tex = open(filepath_tex, 'w')
        fh_tex.write(texcode)
        fh_tex.close()
        
        ## Convert the tex file to a pdf file, and rename/move it to 
        ## the designated directory ('-output-directory=' of pdflatex 
        ## does not seem to work)
        subprocess.call(['pdflatex', filepath_tex])
        subprocess.call(['cp', filepath_pdf, filepath])
        
    
    def to_pickle(self, filepath):
        from cloud.serialization.cloudpickle import dump
        fh = open(filepath, 'w')
        dump(self.__dict__, fh)
        fh.close()


    @staticmethod
    def from_pickle(filepath):
        fh = open(filepath, 'r')
        attrs = cPickle.load(fh)
        fh.close()
        hd = HasseDiagram(hd=attrs)
        return hd
    
    """
    def save(self, attrids=None, filepath):
        # nodes (nodeids, nodeattrs), edges (nodeids), order 
        
        pass
    
    
    @staticmethod
    def load(filepath):
        pass
    """
     
    
    def apply(self, func, print_nodeid=False, pass_exception=False,
              **kwargs):
        """Apply a function to the nodes and modify the HasseDiagram instance
        in-place.
        
        Input:
            func: a function that takes in a tuple of (nodeid, nodeattr) and 
                outputs a new dict
            in_place: 
            print_nodeid: if the application is slow...
            pass_exception:
            kwargs: keyword-arguments for func
        """
        for nodeid in self.nodeids:
            if print_nodeid:
                print nodeid
                
            try:
                nodeattr = self.get_nodeattr(nodeid)
                
                # 1: what if f just modifies existing nodeattrs? Done.
                # 2: it would be more symmetric to return a new tuple (nodeid, nodeattr)... Dismissed. 
                func((nodeid, nodeattr), **kwargs)
            except:
                if pass_exception:
                    pass
                else:
                    raise
            # nodeattrs_new = [f(self.get_nodeattr(nodeid)) for nodeid in self.nodeids]
        # make a new HasseDiagram instance
    
    
    def map(self, func):
        """Apply a function to the nodes and return a new HasseDiagram instance.
        """
        pass
    
    
    def filter(self, f):
        """
        
        Input:
            f: a function that takes in a nodeattr and outputs True or False
        """
        hd_copy = self.copy()
        for rank, nodeids_rank in self.order.items():
            for nodeid in nodeids_rank:
                if not f(self.get_nodeattr(nodeid)):
                    hd_copy.remove_node(nodeid)
                    hd_copy.order[rank].remove(nodeid)
        return hd_copy
    
    
    def merge_duplicate(self):  # ?
        """Useful for creating a diagram from geodesic explorations.
        """
        pass
    
    
    
    def get_f_vector(self):
        return map(len, self.order.values()) + [1]
    
    
    def get_euler(self):
        """Calculate the Euler characteristic. 
        """
        fvec = self.get_f_vector()
        coefs = np.power(-1, range(len(fvec)-2,-2,-1))
        return np.dot(coefs, fvec)
    

    def get_subdiagram(self, root=None, func_node=None, nodeidxs=None,
                       nodeids=None):
        """
        Input:
            root: a nodeid at which the subdiagram is rooted
            func_node: a function that takes in a (nodeid, nodeattr) tuple
                and returns True or False used to decide which nodes are 
                included in the subdiagram
            nodeidxs: a list of node indices to be included in the subdiagram
        """
        if nodeidxs is not None:
            subnodeids = butil.get_subseq(self.nodeids, nodeidxs)
        if nodeids is not None:
            subnodeids = nodeids
        if root:
            subnodeids = [root]
            predecessors = [root]
            while predecessors:
                predecessors_new = []
                for p in predecessors:
                    successors = self.successors(p)
                    subnodeids.extend(successors)
                    predecessors_new.extend(successors)
                predecessors = predecessors_new
        
        rank = max([self.get_nodeattr(nodeid)['rank'] for nodeid in subnodeids])
        subhd = self.subgraph(subnodeids)
        
        # make and attach custom attributes 
        suborder = OD()
        for rank_, nodeids_rank in self.order.items():
            if rank_ <= rank:
                suborder[rank_] = [nodeid for nodeid in nodeids_rank if 
                                   nodeid in subnodeids]    
        subhd.rank = rank
        subhd.order = suborder
        return subhd
    
    
    def change_nodeids(self, mapping):
        """nx.relabel_nodes
        https://networkx.github.io/documentation/development/reference/generated
        /networkx.relabel.relabel_nodes.html#networkx.relabel.relabel_nodes
        """
        pass
    
    
    def get_equivalent_classes(self, f=None):
        """Get the equivalence classes from the given function f. 
        While equivalence classes information can be obvious for small diagrams,
        studying large diagrams require more automated means. 
        Note that it depends on f, that is, data; so an eqcls is not 
        well-defined for a model unless data is also given.  
        
        Input:
            f: a function that takes a nodeid and outputs a list of nodeids
                that are equivalent/symmetric to it; if not given, then each
                node is considered a separate equivalence class 
                (lack of symmetry).
        
        Output:
            eqclss: a mapping from rank to a list of equivalence classes, 
                in the form of lists of nodeids; eg, {1:[[1,2], [3]], 0:[[4]]}
        """
        if f is None:
            f = lambda nodeid: []
            
        eqclss0 = []
        for nodeid in self.nodeids:
            eqcls_ = f(nodeid)  
            eqcls = frozenset([nodeid] + eqcls_)
            eqclss0.append(eqcls)    
        eqclss0 = map(tuple, set(eqclss0))
        
        eqclss = OD([(rank, []) for rank in self.order.keys()])
        for eqcls in eqclss0:
            rank = self.get_nodeattr(eqcls[0])['rank']
            # reordering
            eqcls = [nodeid for nodeid in self.order[rank] if nodeid in eqcls]
            eqclss[rank].append(eqcls)
        
        # put the singletons first
        eqclss = butil.chvals(eqclss, lambda eqclss_: sorted(eqclss_, key=len))
        
        return eqclss
    
    
    @staticmethod
    def _eqclss2eqorder(eqclss):
        """
        Input: 
            eqclss
        
        Output: 
            eqorder: an order that contains equivalence class information; 
                a generalization of order. In each list of rank order, 
                doubletons are symmetric flanking asymmetric nodeids **in
                a list** in the middle. 
        """
        eqorder = OD.fromkeys(eqclss.keys(), None)
        for rank, eqclss_rank in eqclss.items():
            doubletons = [eqcls for eqcls in eqclss_rank if len(eqcls)==2]
            others = [eqcls for eqcls in eqclss_rank if len(eqcls)!=2]
            eqorder[rank] = [butil.flatten(others, 1)]
            for eqcls in doubletons:
                eqorder[rank].append(eqcls[0])
                eqorder[rank].insert(0, eqcls[1])
        return eqorder


    def _get_edge_energy(self, eqorder):
        """
        """
        idxs = butil.flatten(map(range, map(len, eqorder.values())), 1)
        items = zip(butil.flatten(eqorder.values(), 1), idxs)
        nodeid2idx = OD()
        for nodeid, idx in items:
            if isinstance(nodeid, list):
                for _ in nodeid:
                    nodeid2idx[_] = idx
            else:
                nodeid2idx[nodeid] = idx
        energy = sum([abs(nodeid2idx[edgeid[0]]-nodeid2idx[edgeid[1]]) 
                      for edgeid in self.edges()])
        return energy


    def optimize_order(self, eqclss, ngen=10, popsize=5, p=0.5):
        """Optimize the order from the given equivalence class information 
        using evolutionary algorithm, for the purpose of drawing so that
        the edges look less cluttered. 
        
        The optimization is done with an objective function called 
        "edge energy", defined as the sum of node index mismatch for all edges, 
        which is meaningful only for diagrams with **bilateral symmetry**. 
        
        Hence there are two kinds of optimization: 
        1. Swap the nodes in a symmetric pair: 1 ... 2 => 2 ... 1
        2. Reorder the symmetric pairs: 1 3 ... 4 2 => 3 1 ... 2 4
        
        The optimization is done using eqclss, for which the above two 
        operations can be easily done.
        """
        def _mutate_horizontal(eqclss, rank=None, idx=None):
            eqclss = copy.deepcopy(eqclss)
            if rank is None:
                rank = random.sample(eqclss.keys()[1:], 1)[0]
            if idx is None:
                idxs = [i for i,ec in enumerate(eqclss[rank]) if len(ec)>1] 
                if len(idxs) >= 1:
                    idx = random.sample(idxs, 1)[0]
                else:
                    return eqclss
            eqcls = eqclss[rank][idx]
            eqclss[rank][idx] = list(reversed(eqcls))
            return eqclss
            
        def _mutate_vertical(eqclss, rank=None, idx1=None, idx2=None):
            eqclss = copy.deepcopy(eqclss)
            if rank is None:
                rank = random.sample(eqclss.keys()[1:], 1)[0]
            if idx1 is None or idx2 is None:
                idxs = [i for i,ec in enumerate(eqclss[rank]) if len(ec)>1] 
                if len(idxs) >= 2:
                    idx1, idx2 = random.sample(idxs, 2)
                else:
                    return eqclss
            eqcls1 = eqclss[rank][idx1]
            eqcls2 = eqclss[rank][idx2]
            eqclss[rank][idx1] = eqcls2
            eqclss[rank][idx2] = eqcls1
            return eqclss
            
        def _mutate(eqclss):
            if random.random() < p:
                return _mutate_horizontal(eqclss)
            else:
                return _mutate_vertical(eqclss)
        
        for i in range(ngen):
            pop = [eqclss] + [_mutate(eqclss) for j in range(popsize)]
            energies = [self._get_edge_energy(self._eqclss2eqorder(ind)) 
                        for ind in pop]
            idx = energies.index(min(energies))
            eqclss = pop[idx]
        
        return self._eqclss2eqorder(eqclss)
        
    
    def plot(self, nodeid2color=None, theta1s=None, theta2s=None, figsize=None, 
             **kwargs):
        """
        PCA...  Compare different classes/ratelaws...
        """
        assert self.rank == 3  # relax it after coding PCA
        assert 'pred' in self.nodeattrs.columns
        
        if theta1s is None or theta2s is None:
            theta1s = theta2s = np.logspace(-3, 3, 31)
            
        theta1ss, theta2ss = np.meshgrid(theta1s, theta2s)
        thetas = np.logspace(-3, 2, 51)
        
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        ax = plotutil.get_ax3d(figsize)
        
        for nodeid in self.order[2]:
            #ys = np.array([self.get_nodeattr(nodeid, 1).pred((theta1,theta2)) 
            #               for theta1,theta2 in zip(np.ravel(theta1ss), np.ravel(theta2ss))])
            #y1ss = ys[:,0].reshape(theta1ss.shape)
            #y2ss = ys[:,1].reshape(theta1ss.shape)
            #y3ss = ys[:,2].reshape(theta1ss.shape)
            
            #ax.plot_surface(y1ss, y2ss, y3ss, color=nodeid2color[nodeid], alpha=0.3, 
            #                rstride=1, cstride=1, edgecolor='none')
            pred = self.get_nodeattr(nodeid)['pred']
            if nodeid2color is None:
                color = self.get_nodeattr(nodeid)['color']
            else:
                color = nodeid2color[nodeid]
            pred.plot_image(theta1s, theta2s, ax=ax, 
                            color=color,
                            edgecolor='none', **kwargs)
            
            
        for nodeid in self.order[1]:
            if nodeid2color is None:
                color = self.get_nodeattr(nodeid)['color']
            else:
                color = nodeid2color[nodeid]
            ys = np.array([self.get_nodeattr(nodeid, 1).pred([theta]) 
                           for theta in thetas])
            y1s = ys[:,0]
            y2s = ys[:,1]
            y3s = ys[:,2]
            
            ax.plot3D(y1s, y2s, y3s, color=color)
            
        # http://stackoverflow.com/questions/16143493/
        # set-xlim-set-ylim-set-zlim-commands-in-matplotlib-fail-to-clip-displayed-data
        #ax.set_xlim(-50,0)  
        #ax.set_ylim(-50,0)
        #ax.set_zlim(0,50)

        
        plt.show()
        plt.close()

        
        #thetas = np.logspace(-2,2,11)
        #fs = [self.get_nodeattr(nodeid)['pred'] for nodeid in self.order[2]]
        #plotutil.plot_surface(fs,
        #                      thetas, thetas, cs_f=['b','g','r'], alpha=0.2)
                              
        #for rank, nodeids_rank in self.order.items()[1:]:
        #    pass
                 
    
    def get_quotient(self):
        """Return a new HasseDiagram instance with only one member from each 
        equivalence class. 
        https://en.wikipedia.org/wiki/Equivalence_class
        """
        hd = HasseDiagram(rank=self.rank)
        for rank in range(self.rank, -1, -1):
            eqclss_rank = self.eqclss[rank]
            #eqclasses = self.eqcls[rank]
            for eqcls in eqclss_rank:
                hd.add_node(eqcls[0], **self.get_node(eqcls[0]))
        for edgeid in self.edges():
            #nodeid1, nodeid2 = edgeid
            if edgeid[0] in hd.nodeids and edgeid[1] in hd.nodeids:
                hd.add_edge(*edgeid)
        return hd
    
    
    def add_nodeattrs(self, nodeattrs):
        """
        nodeattrs: nodeid2nodeattr
        {'node1': {'attr1':1, 'attr2':2},
         'node2': {'attr1':10, 'attr2':20}}
        
        To save the results of previous time-consuming computation...
        And have a corresponding get_nodeattrs method? From self.nodeattrs?
        
        (Pitfalls of coding:
            Redundancy -> reuse
            Misplaced
            Bad data structure/type)
        """
        pass 
    


def get_product(hds, f_nodeattr=None):
    """Get the product of a list of Hasse diagrams.
    
    Input:
        hds:
        f_nodeattr: a function that takes in a list of nodeattr (dict) from
            the component models and outputs a new dict to be taken as the 
            attributes of the node
            
                
    n=3                                   (a,A)
                            
    n=2                    A              (1,A)  (2,A)  (a,b)  (a,c)
                          / \         
    n=1          a   X   b   c      =     (a,3)  (a,4)  (1,b)  (2,b)  (1,c)  (2,c)
                / \     /     \
    n=0        1   2   3       4          (1,3)  (1,4)  (2,3)  (2,4)
    """
    hd = HasseDiagram(rank=sum([hd_.rank for hd_ in hds]))
    
    nodeinfos = butil.get_product(*[hd_.nodeattrs['rank'].items() for hd_ in hds])
    
    for nodeinfo in nodeinfos:
        nodeid = tuple([tu[0] for tu in nodeinfo])
        rank = sum([tu[1] for tu in nodeinfo])
        if f_nodeattr:
            nodeattr = f_nodeattr([hd_.get_nodeattr(nodeid_)
                                   for hd_, nodeid_ in zip(hds, nodeid)])
        else:
            nodeattr = {}
        hd.add_node(nodeid, rank=rank, **nodeattr)
    
    def _get_edgeids(nodeids, edgeid_, idx):
        if not isinstance(nodeids[0], tuple):
            nodeids = map(lambda nodeid: (nodeid,), nodeids)
        edgeids = [(nodeid[:idx] + (edgeid_[0],) + nodeid[idx:],
                    nodeid[:idx] + (edgeid_[1],) + nodeid[idx:])
                   for nodeid in nodeids]
        return edgeids
    
    edgeids = []
    for idx, hd_ in enumerate(hds):
        nodeids_others = butil.get_product(*[h.nodes() for i, h in enumerate(hds) 
                                             if i!=idx])
        edgeids.extend(butil.flatten([_get_edgeids(nodeids_others, edgeid_, idx) 
                                      for edgeid_ in hd_.edges()], depth=1))
    
    # Note that for-loops are unbearably slow for adding edges
    hd.add_edges_from(edgeids, {'style':'filled'})
    
    return hd    
        

def get_vertex_covers(edges):
    """
    """
    nodeids = list(set(butil.flatten(edges, 1)))
    
    def _work(subset):
        for edge in edges:
            if all([nodeid not in edge for nodeid in subset]):
                return False
        return True

    powerset = butil.powerset(nodeids)
    covers = [subset for subset in powerset if _work(subset)]
    return covers


def is_disjoint(cover, edges):
    """
    """
    for edge in edges:
        if edge[0] in cover and edge[1] in cover:
            return False
    return True



class HasseDiagram0(nx.DiGraph):
    
    def add_node(self, nodeid, level, label='', info='', **kwargs):
        """
        Input:
            nodeid: a str; should be simple, for internal manipulation
            level: an int; depth 
            label: a str; for plotting
            info: a str; should be informative yet consistent, for inspection
            size: 
        """
        if not label:
            label = nodeid
            
        super(HasseDiagram, self).add_node(nodeid, level=level, label=label, 
                                             info=info, **kwargs)
        try:
            self.order[level].append(nodeid)
        except KeyError:
            self.order[level] = [nodeid]
    
    
    def add_edge(self, nodeid1, nodeid2, edgeid=None, info=''):
        """
        Input:
            info: a str; for inspection
        """
        if edgeid is None:
            edgeid = (nodeid1, nodeid2)
        super(HasseDiagram, self).add_edge(nodeid1, nodeid2, edgeid=edgeid, 
                                           info=info)
        

    def reorder(self, level, nodeids):
        """
        Input:
            nodeids: a list of nodeids of the desired order for the given level 
        """
        assert set(nodeids) == set(self.order[level]), "nodeids not right"
        self.order[level] = nodeids
        

    @property
    def nlevel(self):
        return max(self.order.keys())
    
    
    @property
    def nodeids(self):
        #return self.nodes()
        return butil.flatten(self.order.values())

    
    def get_nodeids(self, level=None, f=None):
        """
        Input:
            f: a function that acts on node (with all the attributes)
        """
        if level is not None:
            return self.order[level]
        if f:
            return [nid for nid, node in self.node.items() if f(node)]
    
    
    @property
    def nodeid2info(self):
        infos = [self.node[nodeid]['info'] for nodeid in self.nodeids]
        return OD(zip(self.nodeids, infos))


    @property
    def nodeid2label(self):
        infos = [self.node[nodeid]['label'] for nodeid in self.nodeids]
        return OD(zip(self.nodeids, infos))
        
    
    def set_labels(self, nodeid2label):
        """
        """
        for nid, label in nodeid2label.items():
            self.node[nid]['label'] = label
    change_labels = set_labels  # FIXME ***: deprecation warning
    
    def set_infos(self, nodeid2info):
        """
        """
        for nid, info in nodeid2info.items():
            self.node[nid]['info'] = info
            

            
            
    def merge_duplicates(self, attrname='info'):
        """
        1. Detect duplicates
        2. Remove duplicate nodes
        3. Rewire the edges
        
        Input: 
        """
        for level in range(self.nlevel+1):
            infos_level = set([n['info'] for n in self.node.values() 
                               if n['level']==level])
            for info in infos_level:
                print [nid for nid, n in self.node.items() if n['level']==level and n['info']==info]
                
    
        G.add_node(new_node, attr_dict, **attr) # Add the 'merged' node
        
        for n1,n2,data in G.edges(data=True):
            # For all edges related to one of the nodes to merge,
            # make an edge going to or coming from the `new gene`.
            if n1 in nodes:
                G.add_edge(new_node,n2,data)
            elif n2 in nodes:
                G.add_edge(n1,new_node,data)
        
        for n in nodes: # remove the merged nodes
            G.remove_node(n)
             

    def get_pos(self, height=100, width=100):
        """
        """
        pos = OD()
        for level, nodeids_ in self.order.items():
            for idx, nodeid in enumerate(nodeids_):
                y = (1 - level / self.nlevel) * height
                if level == 0:
                    x = width / 2
                else:
                    x = idx / (len(nodeids_)-1) * width
                pos[nodeid] = (x, y)
        return pos
    
    
    def plot(self, height=100, width=100, crop=True, border=3,
             arrowsize=0.5, nodeshape='box', labelfontsize=8,  
             colornamemap='white', scalemap=0.2, filepath=''):
        """
        Need to add:
            - Fix the box sizes for each level (level2hw)
            - Manually set the edge pos through the box sizes and node pos
        
        Input:
            nodeshape: 'box', 'circle', 'ellipse', etc. 
                (http://www.graphviz.org/doc/info/shapes.html)
            colornamemap: a mapping or a str;
                (http://www.graphviz.org/content/color-names; note that
                 while graphviz supports all the colorname schemes mentioned
                 in the link, pygraphviz seems to support only the X11 scheme.)
            
        """
        import pygraphviz as pgv
        import dot2tex
        
        g = pgv.AGraph(strict=False, directed=True, splines='polyline', 
                       overlap='scale')  # splines='line','false','polyline' all don't seem to work
        g.add_nodes_from(self.nodeids)
        g.add_edges_from(self.edges(), arrowsize=arrowsize)
        
        if isinstance(colornamemap, str):
            colornamemap = dict.fromkeys(self.nodeids, colornamemap)
        if isinstance(scalemap, float):
            scalemap = dict.fromkeys(self.nodeids, scalemap)
        
        pos = self.get_pos(height=height, width=width)
        for nodeid, nodepos in pos.items():
            node = g.get_node(nodeid)
            # graphviz object attributes reference: 
            # http://www.graphviz.org/content/attrs
            node.attr['pos'] = '%f, %f'%tuple(nodepos)
            node.attr['shape'] = nodeshape
            node.attr['fillcolor'] = colornamemap[nodeid]
            node.attr['style'] = 'filled'
            node.attr['size'] = 2.
            node.attr['width'] = scalemap[nodeid]
            node.attr['height'] = scalemap[nodeid]
            node.attr['label'] = self.node[nodeid]['label']
            node.attr['fontsize'] = labelfontsize
        
        for idx, nodeid in enumerate(self.nodeids):
            for nodeid2 in self.neighbors(nodeid):
                g.get_edge(nodeid, nodeid2).attr['sametail'] = str(idx)
                g.get_edge(nodeid, nodeid2).attr['weight'] = 100.  # doesn't seem to work
        
        for idx, nodeid in enumerate(self.nodeids):
            for nodeid0 in self.predecessors(nodeid):
                g.get_edge(nodeid0, nodeid).attr['samehead'] = str(idx + self.number_of_nodes())
        
        #import ipdb
        #ipdb.set_trace()
        
        #g.node_attr['texmode'] = 'math'
        #g.layout(prog='neato')
        
        #g.write('.tmp.dot')
        g.write(path='.tmp.xdot')  #, args='-Gsplines=false')
        fhr = open('.tmp.xdot')
        dotcode = ''.join(fhr.readlines())
        fhr.close()
        
        texcode = dot2tex.dot2tex(dotcode, format='tikz', texmode='math', 
                                  crop=crop)
        if crop:
            texcode = texcode.replace('Border{0pt}', 'Border{%dpt}'%border)
        
        fhw = open(filepath, 'w')
        fhw.write(texcode)
        fhw.close()
        
        #g.draw('tmp/tmp2.dot', prog='neato', args='-n2')
        #g.draw('tmp/tmp3.xdot', prog='neato', args='-n2')
        #g.draw(filepath, prog='neato', args='-n2')
        
    def save(self, filpath):
        raise NotImplementedError
    



def get_hasse_triangle():
    """A triangle
    """
    hd = HasseDiagram(rank=2)
    hd.add_node('F', 0)
    hd.add_node('E1', 1)
    hd.add_node('E2', 1)
    hd.add_node('E3', 1)
    hd.add_node('V1', 2)
    hd.add_node('V2', 2)
    hd.add_node('V3', 2)
    hd.add_edge('F', 'E1')
    hd.add_edge('F', 'E2')
    hd.add_edge('F', 'E3')
    hd.add_edge('E1', 'V1')
    hd.add_edge('E1', 'V2')
    hd.add_edge('E2', 'V2')
    hd.add_edge('E2', 'V3')
    hd.add_edge('E3', 'V3')
    hd.add_edge('E3', 'V1')
    return hd


def get_hasse_rectangle():
    """A rectangle
    """
    hd = HasseDiagram(rank=2)
    hd.add_node('F', 0)
    hd.add_node('E1', 1)
    hd.add_node('E2', 1)
    hd.add_node('E3', 1)
    hd.add_node('E4', 1)
    hd.add_node('V1', 2)
    hd.add_node('V2', 2)
    hd.add_node('V3', 2)
    hd.add_node('V4', 2)
    hd.add_edge('F', 'E1')
    hd.add_edge('F', 'E2')
    hd.add_edge('F', 'E3')
    hd.add_edge('F', 'E4')
    hd.add_edge('E1', 'V1')
    hd.add_edge('E1', 'V2')
    hd.add_edge('E2', 'V2')
    hd.add_edge('E2', 'V3')
    hd.add_edge('E3', 'V3')
    hd.add_edge('E3', 'V4')
    hd.add_edge('E4', 'V4')
    hd.add_edge('E4', 'V1')
    return hd


def get_hasse_tetrahedron():
    """Tetrahedron
    """
    hd = HasseDiagram(rank=3)
    hd.add_node('ABCD', 0)
    hd.add_node('ABC', 1)
    hd.add_node('ABD', 1)
    hd.add_node('ACD', 1)
    hd.add_node('BCD', 1)
    hd.add_node('AB', 2)
    hd.add_node('AC', 2)
    hd.add_node('AD', 2)
    hd.add_node('BC', 2)
    hd.add_node('BD', 2)
    hd.add_node('CD', 2)
    hd.add_node('A', 3)
    hd.add_node('B', 3)
    hd.add_node('C', 3)
    hd.add_node('D', 3)
    hd.add_edge('ABCD', 'ABC')
    hd.add_edge('ABCD', 'ABD')
    hd.add_edge('ABCD', 'ACD')
    hd.add_edge('ABCD', 'BCD')
    hd.add_edge('ABC', 'AB')
    hd.add_edge('ABC', 'AC')
    hd.add_edge('ABC', 'BC')
    hd.add_edge('ABD', 'AB')
    hd.add_edge('ABD', 'AD')
    hd.add_edge('ABD', 'BD')
    hd.add_edge('ACD', 'AC')
    hd.add_edge('ACD', 'AD')
    hd.add_edge('ACD', 'CD')
    hd.add_edge('BCD', 'BC')
    hd.add_edge('BCD', 'BD')
    hd.add_edge('BCD', 'CD')
    hd.add_edge('AB', 'A')
    hd.add_edge('AB', 'B')
    hd.add_edge('AC', 'A')
    hd.add_edge('AC', 'C')
    hd.add_edge('AD', 'A')
    hd.add_edge('AD', 'D')
    hd.add_edge('BC', 'B')
    hd.add_edge('BC', 'C')
    hd.add_edge('BD', 'B')
    hd.add_edge('BD', 'D')
    hd.add_edge('CD', 'C')
    hd.add_edge('CD', 'D')
    return hd


def get_hasse_square_pyramid():
    """Square pyramid
    """
    hd = HasseDiagram(rank=3)
    hd.add_node('ABCDE', 0)
    hd.add_node('ABC', 1)
    hd.add_node('ABE', 1)
    hd.add_node('ABCD', 1)
    hd.add_node('ACD', 1)
    hd.add_node('ADE', 1)
    hd.add_node('AB', 2)
    hd.add_node('AC', 2)
    hd.add_node('BC', 2)
    hd.add_node('CD', 2)
    hd.add_node('BE', 2)
    hd.add_node('DE', 2)
    hd.add_node('AC', 2)
    hd.add_node('AD', 2)
    hd.add_node('AE', 2)
    hd.add_node('B', 3)
    hd.add_node('C', 3)
    hd.add_node('A', 3)
    hd.add_node('D', 3)
    hd.add_node('E', 3)
    hd.add_edge('ABCDE', 'ABC')
    hd.add_edge('ABCDE', 'ACD')
    hd.add_edge('ABCDE', 'ABE')
    hd.add_edge('ABCDE', 'ADE')
    hd.add_edge('ABCDE', 'BCDE')
    
    hd.add_edge('ABC', 'AB')
    hd.add_edge('ABC', 'AC')
    hd.add_edge('ABC', 'BC')
    
    hd.add_edge('ABE', 'AB')
    hd.add_edge('ABE', 'AE')
    hd.add_edge('ABE', 'BE')
    
    hd.add_edge('ACD', 'AC')
    hd.add_edge('ACD', 'AD')
    hd.add_edge('ACD', 'CD')
    
    hd.add_edge('ADE', 'AD')
    hd.add_edge('ADE', 'AE')
    hd.add_edge('ADE', 'DE')
    
    hd.add_edge('BCDE', 'BC')
    hd.add_edge('BCDE', 'BE')
    hd.add_edge('BCDE', 'CE')
    hd.add_edge('BCDE', 'DE')
    
    hd.add_edge('AB', 'A')
    hd.add_edge('AB', 'B')
    hd.add_edge('AC', 'A')
    hd.add_edge('AC', 'C')
    hd.add_edge('AD', 'A')
    hd.add_edge('AD', 'D')
    hd.add_edge('AE', 'A')
    hd.add_edge('AE', 'E')
    
    hd.add_edge('BC', 'B')
    hd.add_edge('BC', 'C')
    hd.add_edge('BE', 'B')
    hd.add_edge('BE', 'E')
    hd.add_edge('CD', 'C')
    hd.add_edge('CD', 'D')
    hd.add_edge('DE', 'D')
    hd.add_edge('DE', 'E')
    
    return hd




"""
from __future__ import division

import numpy as np

from util import plotutil, butil


def plot_isocurve_diagram(bids, xys, pairs, yscale=1, linewidth=1,
                          symmcolor=True, colorscheme='standard',
                          filepath='', show=False, ):

    Input:
        bid2pos: 
        pairs:
        symmcolor: if True, ...; requires the bid in order
        

    #xs = [1,2,3,5,6,7]
    #import ipdb
    #ipdb.set_trace()
    bid2xy = dict(zip(bids, xys))
    xs = [xy[0] for xy in xys]
    
    ax = plotutil.get_axs(1,1, figsize=(6,3), subplots_adjust={'bottom':0.2})[0]
    plotutil.mpl.rc('axes', **{'grid': False})
    
    
    if symmcolor:
        pair2color = {}
        symmpairs = []
        for pair in pairs:
            bid1, bid2 = pair
            bid1_symm = bids[-bids.index(bid1)-1]
            bid2_symm = bids[-bids.index(bid2)-1]
            symmpairs.append(frozenset([pair, (bid2_symm, bid1_symm)]))
        symmpairs_uniq = list(set(symmpairs))
        for idx, symmpair in enumerate(symmpairs_uniq):
            color = plotutil.get_colors(len(symmpairs_uniq), scheme=colorscheme)[idx]
            for pair in symmpair:
                pair2color[pair] = color
        # reorder so that the a symmpair is on top of another symmpair w/o criss-crossing
        pairs = butil.flatten(symmpairs_uniq, depth=1)  
    else:
        colors = plotutil.get_colors(len(pairs), scheme=colorscheme)
        pair2color = dict(zip(pairs, colors))
    
    for pair in pairs:
        bid1, bid2 = pair
        (x1, y1), (x2, y2) = sorted([bid2xy[bid1], bid2xy[bid2]])
        
        assert y1 == y2
        
        r, x_center = (x2 - x1) / 2, (x1 + x2) / 2
        xs_pair = np.linspace(x1, x2, 501)
        ys_pair = np.sqrt(r**2 - (xs_pair - x_center)**2) * yscale + y1
        ax.plot(xs_pair, ys_pair, c=pair2color[pair], linewidth=linewidth)
        ax.set_xticks(xs)
        ax.set_xticklabels(bids)
        ax.set_yticks([])
        ax.set_xlim([min(xs)-0.5, max(xs)+0.5])
        ax.set_ylim([0, (max(xs)-min(xs))/2+0.5])
    
    ax.axis('off')
    
    plotutil.plt.subplots_adjust(**{'top':1, 'bottom':0, 'left':0, 'right':1})
    
    plotutil.plt.savefig(filepath, transparent=True)
    if show:    
        plotutil.plt.show()
    plotutil.plt.close()


    
bids = ['a1','a2','a3','b1','b2','b3']
pairs = [('a1','a2'),
         ('a1','b1'),
         ('a1','b2'),
         ('a1','b3'),
         ('a2','b1'),
         ('a2','b3'),
         ('a3','b2'),
         ('a3','b3'),
         ('b2','b3')]
xys = [(i,2) for i in [1,2,3,5,6,7]]

#plot_isocurve_diagram(bids, xys, pairs, yscale=0.4, filepath='tmp.pdf')

"""

if __name__ == '__main__':
    pass