import methods
from matplotlib import pyplot
from shapely.geometry import *
from descartes.patch import PolygonPatch
from math import atan2, degrees, pi
import itertools
import numpy
import random
import math
import copy,sys,time

import p2t 
import shapely
from scipy.spatial import distance

import kdtree


"""
    
    
"""


class Node:
    def __init__(self,v=None,p=None):
        self.p = p
        self.c = [] #connected to
        self.v = v #store an array of the position

    def distance(self,c):
        return distance.euclidean(self.v,c.v)

    def get2DShapelyPoint(self):
        return Point([self.v[0],self.v[1]])
    def lineParent(self):
        if not self.p is None:  

            return LineString([self.get2DShapelyPoint(),self.p.get2DShapelyPoint()])
        return LineString([self.get2DShapelyPoint(),self.get2DShapelyPoint()])
    def __str__(self):
        return str(self.v)
    def __repr__(self):
        return str(self.v)
    def __getitem__(self,k):
        return self.v[k]
    def __setitem__(self,k,v):
        self.v[k] = v
    def __len__(self):
        return len(self.v)
    def __add__(self,b):
        r = []
        for i in range(len(b)):
            r.append(self[i]+b[i])
        return Node(r)
    def __sub__(self,b):
        r = []
        for i in range(len(b)):
            r.append(self[i]-b[i])
        return Node(r)
    def magnitude(self):
        return self.distance(Node([0 for i in range(len(self))]))

    def normalize(self):
        m = self.magnitude()
        for i in range(len(self)):
            self[i]/=m

    def scalProduct(self,b):
        
        for i in range(len(self)):
            self[i] *= b
        
class Tree:
    #simple data structure stored in an array
    #Should look into quad tree for faster
    def __init__(self):
        self.nodes = []

    def getClosest(self,candidate):
        #LinearSearch
        toReturn = None
        d = "inf"
        for n in self.nodes:
            v = candidate.distance(n)
            #distance is smaller, but as well on top of it. 
            if v<d and candidate.v[2]>n.v[2]:
                d = v
                toReturn = n
        return toReturn

    def addNode(self,n):
        self.nodes.append(n)

class RRT:
    def __init__(self,p,s,t,seed = None,smooth=True,motion=30,max_time = 400, m = 8000):
        #s is the starting position
        #t is the goal position 
        self.s = Node((s[0],s[1],0))
        self.t = t
        self.polygon = p
        
        self.smooth = smooth
        self.motion = motion
        self.max_time = max_time
        self.m = m

        #data structure
        # self.tree = Tree()
        self.tree = kdtree.create(dimensions=3)
        self.tree.add(self.s)
        
        self.outside = multiLevelPoly.bounds
        
        if not seed is None:
            random.seed(seed)
        else:
            self.seed = random.randint(0, sys.maxint)
            random.seed(self.seed)
            
        self.pathsCollection = []

        self.timeDistribution = []
    def reset(self):
        self.tree = kdtree.create(dimensions=3)
        self.tree.add(self.s)

        self.seed = random.randint(0, sys.maxint)
        random.seed(self.seed)
        
        self.tStart = time.clock()

    def path(self,p):
        path = []
        while not p.p is None:
            path.append(p)
            p = p.p
        #Missing the last one
        return [p]+list(reversed(path)) 

    def cleanPath(self,p):
        if p is None:
            return None
        pn = [p[0]]
        i = 1
        
        while not pn[-1] is p[-1]:
            
            segment = LineString([pn[-1].get2DShapelyPoint(),p[i].get2DShapelyPoint()])
            while self.polygon.intersects(segment) is False:
                i+=1
                if i == len(p):
                    # i-=1
                    break
                segment = LineString([pn[-1].get2DShapelyPoint(),p[i].get2DShapelyPoint()])
            i-=1
            pn.append(p[i])
            i+=1

        for i in range(1,len(pn)):
            pn[i].p = pn[i-1]

        return pn


    def search(self):
        for i in range(self.m):
            # p = self.searchValidPoint()
            p = self.searchValidWithMotion()

            if not p is None:
                #Check if near the end
                
                #todo fix the time component for moving as fast as possible. 
                     
                t = Node([self.t[0],self.t[1],p.v[2]])

                if t.distance(p)<100 and \
                        self.polygon.intersects(LineString([t.get2DShapelyPoint(),p.get2DShapelyPoint()])) is False:
                    #if
                    t.v[2] += math.tan(0.11) * t.distance(p)

                    t.p = p
                    if self.smooth is True:
                        return self.cleanPath(self.path(t))
                    else:
                        return self.path(t)

        #return the tree if you did not find anything, might cause problem
        #with the search n paths if uncommon
        #For debug purposes
        # l = list(self.tree.inorder())
        # l = map(lambda x: x.data, l)
        # return l


    def searchValidWithMotion(self):
        # This verifies the segment created after shrinking it
        p = Node([random.randint(self.outside[0],self.outside[2]+1),
            random.randint(self.outside[1],self.outside[3]+1),
            random.randint(0,self.max_time)#this time dimensions in second
            ])
        
        


        pc = self.tree.search_nn(p)[0].data

        if pc is None or pc[2]>p[2]:
            return None



        p[2] = pc[2] + math.tan(0.11) * pc.distance(p)
        

        nr = p - pc 
        nr.normalize()
        nr.scalProduct(self.motion)

        pr = pc + nr
        
        segment = LineString([[pc[0],pc[1]],pr.get2DShapelyPoint()])

        if self.polygon.intersects(segment) is True:
            return None


        pr.p = pc 
        self.tree.add(pr)

        return pr

    def searchValidPoint(self):
        ###This method is not used anymore, using the motion 
        ###one now

        #this RRT is in 3 dimensions, should be looking for n dimensions...
        p = Node([random.randint(self.outside[0],self.outside[2]+1),
            random.randint(self.outside[1],self.outside[3]+1),
            random.randint(0,self.max_time)#this time dimensions in second
            ])
        
        #Check if the point is in a polygon hole
        p_shapely = Point([p.v[0],p.v[1]])
        if self.polygon.intersects(p_shapely) is True:
            return None #not valid point

        #find closest 
        pc = self.tree.search_nn(p)[0].data


        
        if pc is None or pc[2]>p[2]:
            return None

        #Check if the segment is valid
        segment = LineString([[pc[0],pc[1]],p_shapely])

        #Should add the check for velocity
        #Might one to have the player alway walk as fast as possible 
        #simulate player's movement???
        #the value is hard coded and comes from adj = 4.6 and opp = 0.5, 
        # 0.11 = arctan(0.5/4.6) 

        # if math.sin(p.v[2]-pc.v[2]/segment.length)<0.11:
        #     return None

        #Instead of checking the angle, force the solver to always walk
        #as fast as possible. 
        p[2] = pc[2] + math.tan(0.11) * pc.distance(p)

        #Check if the segment is valid
        if self.polygon.intersects(segment) is True:
            return None

        #The motion towards p  
        #Normalize p
        nr = p - pc 
        m_nr = nr.magnitude()
        nr.normalize()
        nr.scalProduct(self.motion)

        #check if adding a too long segment
        if nr.magnitude() > m_nr:
            #then update to the previous length
            nr.normalize()
            nr.scalProduct(m_nr)
        pr = pc + nr
        
        #Check if the segment is valid
        
        pr.p = pc 
        self.tree.add(pr)

        return pr

    def FindNPaths(self,n):

        while len(self.pathsCollection)<n:
            self.reset()
            path = self.search()
            path = self.cleanPath(path)
            # print self.seed
            # print path

            if not path is None:
                self.timeDistribution.append(time.clock()-self.tStart)
                self.pathsCollection.append((self.seed,path))   
                print "(",self.seed,",",path,")"

        # for p in self.pathsCollection:
        #     print p 
        print self.timeDistribution





"""
This is the beginning of the main part
"""

# print sys.argv
args = dict(map(lambda x: x.lstrip('-').split('='),sys.argv[1:]))

"""
args

n   =   is the number of paths to be found
        integer range [0..n]
        default value is 0

name =  is the location of the txt file that represents 
        String where the file is located
        default value is "../wasteland"

smooth = Are the paths at the end smooth or not
         True/False
         default value is True

motion = This the value step in the motion; from q_near to q_rand
         integer >0
         default value is 40

m   =   how many states is the rrt allowed to search
        interge >0
        default value is 8000

s   =   Starting position
        2d array with integer
        default value is [50,350]

t   =   Goal position
        2d array with integer
        default value is [642,517]

seed =  set the seed value for the search algorithm, 
        only work when looking for one path, will be overwrite
        when n>0
        default is 32
        max int value
mtime = The upped bound on time dimension 
        has to be > 0
        default value is 400
"""

#Check the default values in the dict
if not "n" in args:
    args["n"] = 1
if not "name" in args:
    args["name"] = "wasteland"
if not "smooth" in args:
    args["smooth"] = True
if not "motion" in args:
    args["motion"] = 40
if not "m" in args:
    args["m"] = 8000

if not "s" in args:
    args["s"] = [50,350]
else:
    args["s"] = eval(args["s"])

if not "t" in args:
    args["t"] = [642,517]
else:
    args["t"] = eval(args["t"])

if not "seed" in args:
    args["seed"] = 32

if not "mtime" in args:
    args["mtime"] = 400


#Load the geometries
multiLevelPoly = methods.getLevelPolygon(args["name"])


rrt = RRT(multiLevelPoly,args["s"],args["t"],int(args["seed"]),\
    smooth = bool(args["smooth"]),motion=int(args["motion"]),max_time=int(args["mtime"]), m=int(args["m"]))


path = None

if args["n"]>1:
    rrt.FindNPaths(int(args["n"]))
else:
    path = rrt.search()


if path is None:
    exit()









outside = multiLevelPoly.bounds
outside = Polygon([(outside[0],outside[1]),
    (outside[0],outside[3]),
    (outside[2],outside[3]),
    (outside[2],outside[1])])


#ploting
fig = pyplot.figure(1)
ax = fig.add_subplot(111)


for p in multiLevelPoly:
    patch = PolygonPatch(p, facecolor=methods.v_color(p), edgecolor=methods.v_color(p), alpha=0.5, zorder=2)
    ax.add_patch(patch)


ax.text(args["s"][0]+7,args["s"][1]+15, r'$\sigma_{\mathit{init}}$', fontsize=13)
methods.plot_coords(ax,Point(args["s"]),c = "#00FF00",z=6)

ax.text(args["t"][0]-30,args["t"][1]+11, r'$\Sigma_{\mathit{goal}}$', fontsize=13)
methods.plot_coords(ax,Point(args["t"]),c = "#FF0000",z=6)

if rrt.smooth is False:
    for p in path:
        methods.plot_coords(ax,p.get2DShapelyPoint(),c = "#0000AA",a=0.5,z=4)
        methods.plot_line(ax,p.lineParent(),w=2,c = "#0000AA",z=5,a=0.4)

    path = rrt.cleanPath(path)

    for p in path:
        methods.plot_coords(ax,p.get2DShapelyPoint(),c = "#00FF00",z=5)
        methods.plot_line(ax,p.lineParent(),c="#00FF00",a=0.7)
    
else:
    for p in path:
        methods.plot_coords(ax,p.get2DShapelyPoint(),c = "#FF0000",z=4)
        methods.plot_line(ax,p.lineParent())

xrange = [outside.bounds[0], outside.bounds[2]]
yrange = [outside.bounds[1], outside.bounds[3]]
ax.set_xlim(*xrange)
ax.set_ylim(*yrange)
ax.set_aspect(1)
ax.invert_yaxis()

ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
pyplot.show()
# pyplot.savefig("result_search.pdf",bbox_inches='tight')





