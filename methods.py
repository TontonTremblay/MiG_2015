from matplotlib import pyplot
# from shapely.geometry import Polygon
from shapely.geometry import *
# from shapely.geometry import LineString
from descartes.patch import PolygonPatch
from math import atan2, degrees, pi
import itertools
import numpy
# from scipy.spatial import Delaunay
import random
import math
import copy

import p2t 

import shapely
COLOR = {
    True:  '#000000',
    False: '#ff3333'
    }

def v_color(ob):
    return COLOR[ob.is_valid]

def plot_coords(ax, ob, c = "#999999",z=3,a = 1):
    x, y = ob.xy
    ax.plot(x, y, 'o', color=c, zorder=z,alpha=a)
  
def plot_bounds(ax, ob):
    x, y = zip(*list((p.x, p.y) for p in ob.boundary))
    ax.plot(x, y, 'o', color='#000000', zorder=1)

def plot_line(ax, ob,c = "#8251f1",a=0.7,w = 3,z=2):
    x, y = ob.xy
    # ax.plot(x, y, color=v_color(ob), alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
    ax.plot(x, y, color=c, alpha=a, linewidth=w, solid_capstyle='round', zorder=z)

def drawTriangle(ax,tri, id = None):
    
    plot_line(ax, LineString( [(tri.a.x,tri.a.y),(tri.b.x,tri.b.y)]),a = 1,c = "#CCCCCC",w =1)
    plot_line(ax, LineString( [(tri.a.x,tri.a.y),(tri.c.x,tri.c.y)]),a = 1,c = "#CCCCCC",w =1)
    plot_line(ax, LineString( [(tri.b.x,tri.b.y),(tri.c.x,tri.c.y)]),a = 1,c = "#CCCCCC",w =1)

    if not id is None:
        p = centerTriangle( arrayPoint( tri ))
        ax.text(p[0]+1, p[1]+1, id, fontsize=12)


def plot_multiLines(ax,ob,c ="#8251f1",a=0.7 ):
	for i in ob:
		plot_line(ax,i,c,a)

def plot_points(ax,ob,c="#999999"):
    x = [] 
    y = []
    for i in ob:
        x.append(i.x)
        y.append(i.y)
    
    ax.plot(x, y, 'o', color=c, zorder=3)

def plot_point(ax,ob,c="#999999",z = 3):
    # print c
    ax.plot(ob[0],ob[1],'o',color =c, zorder = z)

def plot_CoordsList(ax,xy,c ='#999999',a=1):
   
    x = map(lambda x: x[0],xy)
    y = map(lambda v: v[1],xy)
    ax.plot(list(x), list(y), 'o', color=c,alpha=a, zorder=3)

def plot_PointList(ax,xy,c ='#999999',a=1):
    
    x = map(lambda x: x[0],xy)
    y = map(lambda v: v[1],xy)
    ax.plot(list(x), list(y), 'o', color=c,alpha=a, zorder=3)




def angle(origin,pos):
    dx = pos[0] - origin[0]
    dy = pos[1] - origin[1]
    rads = atan2(-dy,dx)
    rads %= 2*pi
    degs = degrees(rads)
    return degs

def distance(v1,v2):
    return math.sqrt( (v1[0]-v2[0])**2 + (v1[1] - v2[1])**2)

def add(v1,v2):
    return (v1[0]+v2[0],v1[1]+v2[1])
def substract(v1,v2):
    return (v1[0]-v2[0],v1[1]-v2[1])
def magnitude(v):
    return math.sqrt(v[0]*v[0]+v[1]**2)
def normalize(v):

    return (v[0]/magnitude(v),v[1]/magnitude(v))

def getPointsExport(p):
    s = str(p)

    # print str(p.geom_type)
    if str(p.geom_type) == "Polygon" and not list(p.interiors) is []:
        
        polys = []
        polys.append(getPoints(p.exterior))
        for inter in p.interiors:
            polys.append(getPoints(inter))
                
        return polys 
    else:
        r = s.replace("EMPTY","").replace("LINEARRING ","").replace("POLYGON ","").replace("MULTIPOLYGON ","").replace("MULTILINESTRING ","").replace("LINESTRING ","").replace("MULTIPOINT ","").replace("POINT ","").replace("GEOMETRYCOLLECTION ","").replace("(","").replace(")","").replace(", ",",")

        r = r.split(",")
        r = [i.split(" ") for i in r]

        r = list(map(lambda x: (float(x[0]),float(x[1])),r))
        return r

def getPoints(p):
    s = str(p)


    r = s.replace("EMPTY","").replace("LINEARRING ","").replace("POLYGON ","").replace("MULTIPOLYGON ","").replace("MULTILINESTRING ","").replace("LINESTRING ","").replace("MULTIPOINT ","").replace("POINT ","").replace("GEOMETRYCOLLECTION ","").replace("(","").replace(")","").replace(", ",",")

    r = r.split(",")
    r = [i.split(" ") for i in r]

    r = list(map(lambda x: (float(x[0]),float(x[1])),r))
    return r

def scalarProduct(v,s):
    return (v[0]*s,v[1]*s)

def neg(v):
    return (-v[0],-v[1])

def rotationPolygon(poly,theta):
    theta = -theta
    polyToReturn = []
    points = getPoints(poly)

    for i in points:
        x,y = i
        temp = (x*math.cos(math.radians(theta)) - y*math.sin(math.radians(theta)),x*math.sin(math.radians(theta))+y*math.cos(math.radians(theta)))
        polyToReturn.append(temp)
    return Polygon(polyToReturn)

def translationPolygon(poly,v):
    polyToReturn = []
    points = getPoints(poly)

    for i in points:
        x,y = i
        temp = (x+v[0],y+v[1])
        polyToReturn.append(temp)
    
    return Polygon(polyToReturn)



def pointIsInArray(a,p):
    e = 0.001
    for i in a:
        if distance(i,p)<e:
            return True
    return False

def savedata(func):
    d = dict() 


    def vision(polyLevel,p1,ax = None):
        pValue = getPoints(p1)[0]       
        if not pValue in d:
            d[pValue] = func(polyLevel,p1,ax)
            
        return d[pValue]
        
    return vision





def savedataFromTriangle(func):
    d = dict() 
    triangles = []

    def inner(*args,**kargs):
        if "data" in kargs:
            for k in kargs["data"].keys():
                d[k] = kargs["data"][k]
            return

        if "triangle" in kargs:
            for t in list(kargs["triangle"]):            
                triangles.append(t)
            return 
    
        pValue = getPoints(args[1])[0]

        #Find the triangle the point is in
        for i in triangles:

            if args[1].within( Polygon(arrayPoint(i))):
                pValue = centerTriangle (i)

                break
        if not pValue in d:
            
            d[pValue] = func(*args,**kargs)
            
        return d[pValue]

    return inner


@savedataFromTriangle
def GetVisionPolygon(polyLevel,p1,ax = None):
    """ This method returns the visibility polygon of 
        one point. 
    """
    #######################
    # GET THE OUTLINE COORDINATES

    multiLevelPoly = MultiPolygon(polyLevel)
    multiLevelPoly = multiLevelPoly.union(multiLevelPoly)

    # could not find a simple way to get only the points so use the toString. 
    # This is a little bit of a hack

    levelPoints = []
    for p in polyLevel:
        s = str(p)
        r = s.replace("POLYGON ","").replace(", ",",").replace("(","").replace(")","")
        r = r.split(",")
        r = [i.split(" ") for i in r]
        # print r
        r = list(set(map(lambda x: (float(x[0]),float(x[1])),r)))
        levelPoints+= r


    # remove the outside from the level
    # levelPoints = levelPoints - outsidePoints
        

    # get a line from the first point on the roadmap to all the points. 

    vision = []


    for i in levelPoints:
        p = p1.coords[0]
        dirVector = normalize( substract(i,p) )
        dirVector = scalarProduct(dirVector,50)
        # print substract(i,p), dirVector, add(i,dirVector)

        vision.append(LineString([p,i]))


    # Sort the rays by angle first. 
    raysSorted = [ ]

    for r in vision:
        # print vreference
        raysSorted.append( (-angle(r.coords[0],r.coords[1]), r))

    raysSorted = sorted(raysSorted)
    raysSorted = [(0,LineString( (p,(p[0]+1000,p[1]) ) ))] + raysSorted



    # Found all the intersections from line 
    interPoints = dict()

    allIntersecPoints = []
    polyJustAdded = -1#This keep track on which polygon we are working on. 

    for i,line in enumerate(raysSorted):
        idValue = i
        #same hack to get the points
        # print multiLevelPoly.is_valid
        # change for each polygon
        r1 = "" 
        collision = line[1].intersection(multiLevelPoly)
        r1 = str(collision)
        r1 = r1.replace("EMPTY","").replace("MULTIPOLYGON ","").replace("MULTILINESTRING ","").replace("LINESTRING ","").replace("MULTIPOINT ","").replace("POINT ","").replace("GEOMETRYCOLLECTION ","").replace("(","").replace(")","").replace(", ",",")
        r1 = r1.split(',')
        r1 = [x.split(" ") for x in r1]   
        r1 = list(set(map(lambda x: (float(x[0]),float(x[1])),r1)))
        r = sorted(r1, key=lambda v: distance(p,v))
        
        if i == 0:
            allIntersecPoints.append(r[0])
            if not ax is None:
                plot_point(ax,r[0],c = "#999999")
        


        if r[0] in levelPoints and not r[0] in allIntersecPoints:
            # Send an other ray
            # other ray
            i = r[0]
            p = p1.coords[0]
            dirVector = normalize( substract(i,p) )

            pNear = add(i , dirVector )
            dirVector = scalarProduct(dirVector,10000)
            pFar = add(i , dirVector)

            if "EMPTY" in str(Point(pNear).intersection(multiLevelPoly)):

                line =  LineString([pNear,pFar])
                collision = line.intersection(multiLevelPoly)
                r1 = str(collision)
                r1 = r1.replace("EMPTY","").replace("MULTIPOLYGON ","").replace("MULTILINESTRING ","").replace("LINESTRING ","").replace("MULTIPOINT ","").replace("POINT ","").replace("GEOMETRYCOLLECTION ","").replace("(","").replace(")","").replace(", ",",")
                r1 = r1.split(',')
                r1 = [x.split(" ") for x in r1]   
                r1 = list(set(map(lambda x: (float(x[0]),float(x[1])),r1)))
                r2 = sorted(r1, key=lambda v: distance(p,v))
                
                # if idValue == 99:
                #     ax.text(r[0][0]+1, r[0][1]+1, str("r"), fontsize=12)
                #     ax.text(r2[0][0]+1, r2[0][1]+1, str("r2"), fontsize=12)

                #     plot_line(ax,LineString([r[0],allIntersecPoints[-1]]) ,c = "#0ACC0A")
                #     plot_line(ax,LineString([r2[0],allIntersecPoints[-1]]),c = "#CC0ACC")

                #     print "intersections"
                #     print "r",LineString([r[0],allIntersecPoints[-1]]).intersection(multiLevelPoly)
                #     print "r2",LineString([r2[0],allIntersecPoints[-1]]).intersection(multiLevelPoly)

                #     print "touches"
                #     print "r",LineString([r[0],allIntersecPoints[-1]]).touches(multiLevelPoly)
                #     print "r2",LineString([r2[0],allIntersecPoints[-1]]).touches(multiLevelPoly)

                #     print "crosses"
                #     print "r",LineString([r[0],allIntersecPoints[-1]]).crosses(multiLevelPoly)
                #     print "r2",LineString([r2[0],allIntersecPoints[-1]]).crosses(multiLevelPoly)
                    
                #     print "within"
                #     print "r",LineString([r[0],allIntersecPoints[-1]]).within(multiLevelPoly)
                #     print "r2",LineString([r2[0],allIntersecPoints[-1]]).within(multiLevelPoly)
                 

                if "LINESTRING" in str(LineString([r[0],allIntersecPoints[-1]]).intersection(multiLevelPoly)):
                # if LineString([r[0],allIntersecPoints[-1]]).touches(multiLevelPoly) and \
                #     not LineString([r[0],allIntersecPoints[-1]]).crosses(multiLevelPoly):       
                    # plot_PointList(ax,r2,c="#0000FF")
                    allIntersecPoints.append(r[0])
                    allIntersecPoints.append(r2[0])
                else:

                    # if not LineString([r2[0],allIntersecPoints[-1]]).within(multiLevelPoly) and \
                    #    LineString([r2[0],allIntersecPoints[-1]]).touches(multiLevelPoly) :

                    allIntersecPoints.append(r2[0])
                    allIntersecPoints.append(r[0])
                    
                    # else:
                    #     allIntersecPoints.append(r[0])
                    #     allIntersecPoints.append(r2[0])   

                if not ax is None:  
                    plot_point(ax,r[0],c = "#FF4A71")
                    plot_point(ax,r2[0],c = "#FFB375")
                    # ax.text(r[0][0]+2,r[0][1]+2,str(idValue))


            else:

                allIntersecPoints.append(r[0])
                
                if not ax is None:
                    plot_point(ax,r[0],c = "#75C1FF")
                    # ax.text(r[0][0]+2,r[0][1]+2,str(idValue))


    # print allIntersecPoints

    # if not ax is None:
    #     plot_PointList(ax,allIntersecPoints)





    # create polygon 
    visionPoly = None
    visionPoly = Polygon( [i for i in allIntersecPoints] )

    if not ax is None:
        patch = PolygonPatch(visionPoly, facecolor="#FF00FF", edgecolor=v_color(visionPoly), alpha=0.1, zorder=2)
        ax.add_patch(patch)
    return visionPoly




def linePointsBetween(v1,v2,d):
    """ this goes from v1 to v2 with 
        interval of d. Will include v1 and v2 
    """
    toReturn = [v1]
    dirVector = normalize( substract(v2,v1) )
    dirVector = scalarProduct(dirVector,d)
    atNow = copy.copy(v1)
    distMax = distance(v1,v2)
    distDone = 0 
    while distDone+d<distMax:
        atNow = add(atNow,dirVector)
        toReturn.append(atNow)
        distDone+=d

    toReturn.append(v2)
    return toReturn


def OneFovFromPoint(multiLevelPoly,p,theta = 0):
    dist_fov = 50
    angle_fov = 15 #this is in degree. 

    fovPoly = Polygon([(0,0),(-dist_fov * math.cos(angle_fov), dist_fov * math.sin(angle_fov)),(dist_fov,0),
        (-dist_fov * math.cos(angle_fov), -dist_fov * math.sin(angle_fov))])

    finalVisionPoly = Polygon()

    fovPoly = rotationPolygon(fovPoly,theta)
    fovPoly = translationPolygon(fovPoly, p)
    
    visionPoly = GetVisionPolygon(multiLevelPoly, Point(p))
    finalVisionPoly = finalVisionPoly.union (fovPoly.intersection(visionPoly))
    return finalVisionPoly



def OnePathVision(multiLevelPoly,path,dist = 10,simplyFactor=5):
    """ This method returns the visibility for one region
    """
    dist_fov = 50
    angle_fov = 15 #this is in degree. 

    fovPoly = Polygon([(0,0),(-dist_fov * math.cos(angle_fov), dist_fov * math.sin(angle_fov)),(dist_fov,0),
        (-dist_fov * math.cos(angle_fov), -dist_fov * math.sin(angle_fov))])

    finalVisionPoly = Polygon()

    # fov1 = Polygon([(0,0),(-dist_fov * math.cos(angle_fov), dist_fov * math.sin(angle_fov)),(dist_fov,0),
    #     (-dist_fov * math.cos(angle_fov), -dist_fov * math.sin(angle_fov))])
    # fov2 = Polygon([(0,0),(-dist_fov * math.cos(angle_fov), dist_fov * math.sin(angle_fov)),(dist_fov,0),
    #     (-dist_fov * math.cos(angle_fov), -dist_fov * math.sin(angle_fov))])




    points = getPoints(path)
    for i in range(0,len(points)-1):
        toEvaluate = linePointsBetween(points[i],points[i+1],dist)
        fovPoly = rotationPolygon(fovPoly, angle(points[i],points[i+1]))


        for j,p in enumerate(toEvaluate):
            fovPoly = translationPolygon(fovPoly, p)
            visionPoly = GetVisionPolygon(multiLevelPoly,Point(p))
            finalVisionPoly = finalVisionPoly.union (fovPoly.intersection(visionPoly))
            

            fovPoly = translationPolygon(fovPoly, neg(p))
        

        # fov1 = rotationPolygon(fov1, angle(points[i],points[i+1]))
        # fov2 = rotationPolygon(fov2, angle(points[i],points[i+1]))

        # fov1 = translationPolygon(fov1,points[i])
        # fov2 = translationPolygon(fov2,points[i+1])

        # fovPoly = MultiPolygon([fov1,fov2]).convex_hull

        
        # visionPoly = GetVisionPolygon(multiLevelPoly,Point(points[i]))
        
        # finalVisionPoly = finalVisionPoly.union (fovPoly.intersection(visionPoly))
        
        # fov1 = translationPolygon(fov1, neg(points[i]))
        # fov2 = translationPolygon(fov2, neg(points[i+1]))
        
        # fov1 = rotationPolygon(fov1, - angle(points[i],points[i+1]))
        # fov2 = rotationPolygon(fov2, - angle(points[i],points[i+1]))

        # fovPoly = rotationPolygon(fovPoly, - angle(points[i],points[i+1]))
    
    # finalVisionPoly = finalVisionPoly.simplify(simplyFactor)

    # print finalVisionPoly
    return finalVisionPoly



def getFOV(typeFOV):
    dist_fov = 50

    if typeFOV is "triangle":
        angle_fov = 15 #this is in degree. 

        return Polygon([(0,0),(-dist_fov * math.cos(angle_fov), dist_fov * math.sin(angle_fov)),(dist_fov,0),
            (-dist_fov * math.cos(angle_fov), -dist_fov * math.sin(angle_fov))])
    else:

        return Point([0,0]).buffer(dist_fov)


def getPolyVisionOld(multiLevelPoly,pathsToEnd,dist =10,simplyFactor=5):
    """ This is the method returns the collection vision regions
        This method is old and I am keeping it only for testing.
    """

    dist_fov = 50
    angle_fov = 15 #this is in degree. 

    fovPoly = Polygon([(0,0),(-dist_fov * math.cos(angle_fov), dist_fov * math.sin(angle_fov)),(dist_fov,0),
        (-dist_fov * math.cos(angle_fov), -dist_fov * math.sin(angle_fov))])



    polyPathsVision = []
    for path in pathsToEnd:

        finalVisionPoly = Polygon()



        points = getPoints(path)
        for i in range(0,len(points)-1):
            toEvaluate = linePointsBetween(points[i],points[i+1],dist)
            fovPoly = rotationPolygon(fovPoly, angle(points[i],points[i+1]))
            

            for j,p in enumerate(toEvaluate):
                fovPoly = translationPolygon(fovPoly, p)
                visionPoly = GetVisionPolygon(multiLevelPoly,Point(p))
                finalVisionPoly = finalVisionPoly.union (fovPoly.intersection(visionPoly))
                fovPoly = translationPolygon(fovPoly, neg(p))
            
            fovPoly = rotationPolygon(fovPoly, - angle(points[i],points[i+1]))
        

        polyPathsVision.append(finalVisionPoly)

    return polyPathsVision

def getPolyVision(multiLevelPoly,pathsToEnd,dist =10,simplyFactor=5):
    """ This is the method returns the collection vision regions
    """


    t = "circle"
    fov1 = getFOV(t)
    fov2 = getFOV(t)

    polyPathsVision = []
    for path in pathsToEnd:

        finalVisionPoly = Polygon()



        points = getPoints(path)
        for i in range(0,len(points)-1):
            fov1 = rotationPolygon(fov1, angle(points[i],points[i+1]))
            fov2 = rotationPolygon(fov2, angle(points[i],points[i+1]))

            fov1 = translationPolygon(fov1,points[i])
            fov2 = translationPolygon(fov2,points[i+1])

            fovPoly = MultiPolygon([fov1,fov2]).convex_hull

            
            visionPoly = GetVisionPolygon(multiLevelPoly,Point(points[i]))
            
            finalVisionPoly = finalVisionPoly.union (fovPoly.intersection(visionPoly))
            
            fov1 = translationPolygon(fov1, neg(points[i]))
            fov2 = translationPolygon(fov2, neg(points[i+1]))
            
            fov1 = rotationPolygon(fov1, - angle(points[i],points[i+1]))
            fov2 = rotationPolygon(fov2, - angle(points[i],points[i+1]))
        
        # finalVisionPoly = finalVisionPoly.simplify(simplyFactor)
        
        polyPathsVision.append(finalVisionPoly)

    return polyPathsVision


class Node:
    def __init__(self,id,pos = (0,0),a = []):
        self.neighbours = a 
        self.id = id
        self.pos = pos

    def __str__(self):
        return str(self.id) + " " +str(self.pos) +  " " +str(self.neighbours)

class Graph:
    def __init__(self,fileName = None):
        self.nodes = [] 
        self.simplePaths = []
        if not fileName is None:
            self.create(fileName) 
    def createFromTriangle(self,data):
        for i in data :
            self.nodes.append(Node(i[0],i[1],i[2]))

    def create(self,fileName="graphtest.txt"):
        data = open(fileName)
        graphInfo = []
        #Create the nodes
        for i in data  :
            i = eval(i)
            self.nodes.append(Node(i[0],i[1],i[2]))
        data.close()

        
        #initialize the neighboors
        for i in graphInfo:
            self.nodes[i[0]].setNeighbours
    
    def display(self,ax = None):
        if ax is None:
            return
        for i in self.nodes:
            plot_coords(ax,Point(i.pos),c = "#00FF00")
            ax.text(i.pos[0]+1, i.pos[1]+1, i.id, fontsize=10)
            for j in i.neighbours:
                plot_line(ax,LineString([i.pos,self.nodes[j].pos]))



    def findSimplePath(self,s,t):
        """ This method finds the simple path from s to t. 
            it is using the node id. 
        """
        self.simplePaths = []

        if s == t:
            return []


        used = [0 for i in range(len(self.nodes))]
        n = s
        used[n] = 1
        sol = [s]

        self.depthFirstSearch(n,t,copy.copy(used),sol)
        return self.simplePaths
        

    def depthFirstSearch(self,n,t,nodeSearched,sol):
        
        if n == t:
            return sol

        a = []

        for i in self.nodes[n].neighbours:
            if nodeSearched[i] is 0: 

                newNodes = copy.copy(nodeSearched)
                newNodes[i] = 1
                newSol = copy.copy(sol)
                newSol.append(i)
                newSol = self.depthFirstSearch(i,t,newNodes,newSol)
                if not newSol is None:
                    self.simplePaths.append( newSol)
        



    def __str__(self):
        s = ""
        for i in self.nodes:
            s+= str(i) + "\n"
        return s


def arrayPoint(tri):
    return [[tri.a.x,tri.a.y],[tri.b.x,tri.b.y],[tri.c.x,tri.c.y]]

def centerTriangle(a1):
    if a1.__class__ is p2t.Triangle:
        a1 = arrayPoint(a1)
    x = 0.0
    y = 0.0
    for i in a1:
        x+= i[0]
        y += i[1]
    return (x/3,y/3)

def midPoint(p1,p2):
    return ((p1[0]+p2[0])/2,(p1[1]+p2[1])/2)

def checkPairs(p1,p2):
    
    return p1 == p2 
    
def shareEdge(t1,t2):

    t1 = arrayPoint(t1)
    t2 = arrayPoint(t2)
    e1 = list(itertools.combinations(t1,2))
    e2 = list(itertools.combinations(t2,2))

    for i in itertools.product(e1,e2):
        if  checkPairs (i[0], i[1]) or checkPairs (i[0], tuple(reversed(i[1]))) :
            
            return LineString([centerTriangle(t1), midPoint(i[0][0],i[0][1])  ,centerTriangle(t2)])
    
    return None

def mainTestGraph():
    g = Graph()
    g.create("level1graph.txt")
    print g

    fig = pyplot.figure(1)
    ax = fig.add_subplot(111)

    g.display(ax)
    pyplot.show()

    print    g.findSimplePath(0,4)

def getPolygonWithHoles(p):
    extLevel = []
    intLevel = []
    p = MultiPolygon(p)
    for i in p:
        v = list(i.interiors)

        if len(v) == 0:
            intLevel.append( [p2t.Point(p[0],p[1]) for p in getPoints(i.exterior)[:-1]])
        else:            
            for j in v:
                points = getPoints(j)[:-1]
                extLevel +=  [p2t.Point(p[0],p[1]) for p in points]
    return (extLevel,intLevel)

def SaveMethodData(func):
    data = []
    def inner(*args,**kargs):
        if len(data)>0:
            return data
        else:
            r = func(*args,**kargs)
            for v in r:
                data.append(v)
            return r
    return inner

@SaveMethodData
def TriangulatePolygon(p):
    extLevel, intLevel = getPolygonWithHoles(p)
    cdt = p2t.CDT(extLevel)

    for i in intLevel:
        #Create the holes
        cdt.add_hole(i)
        
    return cdt.triangulate()
    

def getTriangleIndex(triangles, p):
    for i in range(len(triangles)):
        if Point(p).within( Polygon(arrayPoint( triangles[i] ))):
            return i
    return None

def getShareEdge(t1,t2):
    t1 = arrayPoint(t1)
    t2 = arrayPoint(t2)
    e1 = list(itertools.combinations(t1,2))
    e2 = list(itertools.combinations(t2,2))

    for i in itertools.product(e1,e2):
        if  checkPairs (i[0], i[1]) or checkPairs (i[0], tuple(reversed(i[1]))) :
            
            return (i[0][0],i[0][1])
    


def testTriangulationPolygon():
    name = "wasteland"
    # name = "leveltest"
    #Load the geometries
    multiLevelPoly = getLevelPolygon("wasteland")




    outside = multiLevelPoly.bounds
    outside = Polygon([(outside[0],outside[1]),(outside[0],outside[3]),(outside[2],outside[3]),(outside[2],outside[1])])

    #This gives the definition of the interior polygon
    #From there we can do a triangulation
    #Hummmm
   
    

    #Draw triangulation
    fig = pyplot.figure(1)
    ax = fig.add_subplot(111)


    for p in multiLevelPoly:
       patch = PolygonPatch(p, facecolor=v_color(p), edgecolor=v_color(p), alpha=0.5, zorder=2)
       ax.add_patch(patch)


    tri = TriangulatePolygon(multiLevelPoly)
    for i in range(len(tri)):
        # drawTriangle(ax,tri[i],id=i)
        drawTriangle(ax,tri[i])

    neighbours = [[] for i in range(len(tri))]
    for i in range(len(tri)):
        for j in range(i+1,len(tri)):
            r = shareEdge(tri[i],tri[j])
            if not r is None:   
                # plot_line(ax,r,w=1) 

                neighbours[i].append(j)
                neighbours[j].append(i)
    
    data = [ [i,(centerTriangle(tri[i])),neighbours[i] ] for i in range(len(tri))]

    g = Graph()
    g.createFromTriangle(data)
    
    paths =  g.findSimplePath(7,1)
    print paths

    #Find full path from the road map triangulation. 


    #Get the path 
    pathsToEnd = []
    for p in paths:
        line = [centerTriangle(tri[p[0]])]
        for i in range(1,len(p)-1):
            edge = getShareEdge(tri[p[i]],tri[p[i+1]])
            line += [midPoint(edge[0],edge[1]), centerTriangle(tri[p[i+1]])] 
        print line
        pathsToEnd.append(LineString(line))
    
    level = open(name+"visionTri.txt")
    polyPathsVision = []
    for i in level:
        line = eval(i)
        polyPathsVision.append(Polygon(line))

    if len(polyPathsVision) == 0:

        polyPathsVision = getPolyVision(multiLevelPoly, pathsToEnd)
        
        # Write to the file.
        s = ""     
        for i in polyPathsVision:
            s+= str(getPoints(i)) + "\n"
        level = open(name+"visionTri.txt",'w')
        level.write(s)
        level.close()
    visionRegion = polyPathsVision[0]
    patch = PolygonPatch(visionRegion, facecolor="#FF00FF", edgecolor=v_color(visionRegion), alpha=0.1, zorder=1)
    ax.add_patch(patch)

    #Construct the graph data structure from the triangles. 
    #Add their id as welll 

    xrange = [outside.bounds[0], outside.bounds[2]]
    yrange = [outside.bounds[1], outside.bounds[3]]
    ax.set_xlim(*xrange)
    # ax.set_xticks(range(xrange[0],xrange[-1],100) )
    ax.set_ylim(*yrange)
    # ax.set_yticks(range(yrange[0],yrange[-1],100) )
    ax.set_aspect(1)
    ax.invert_yaxis()
    pyplot.show()




def getPathsFromTo(multiLevelPoly, name, s, t,visionCalculated=True,ax=None):
    """ This method also computes the vision for each triangle in 
        the level and then memoizates it using a decorator.  
    """

    tri = TriangulatePolygon(multiLevelPoly)
    #Set up the vision for each triangles, this make the assumption 
    #that the triangles are pretty close to each other.

    if not isinstance(s,int):
        #find the index 
        s = getTriangleIndex(tri,Point(s))
        t = getTriangleIndex(tri,Point(t))

    if not ax is None:
        for i in range(len(tri)):
            drawTriangle(ax,tri[i])

    data = list(open(name+"visionTri.txt"))

    toSave = []
    GetVisionPolygon(triangle = tri)

    if visionCalculated:
        if len(data) == 0:
            #get the values
            
            for i in range(len(tri)):
            
                # drawTriangle(ax,tri[i],id=i)
                point = centerTriangle(tri[i])
                toSave.append( (centerTriangle(tri[i]), getPoints (GetVisionPolygon(multiLevelPoly,Point(point)) )  ) )
            
            #save the data calculated
            toSaveFile = open(name+"visionTri.txt","w")
            for i in toSave:
                toSaveFile.write(str(i)+"\n")
            toSaveFile.close()

        else:
            d = dict()
            for i in data:
                i = eval(i)

                d[i[0]] = Polygon(i[1])    
            GetVisionPolygon(data=d)   #decorator set the polyvision data
            
    neighbours = [[] for i in range(len(tri))]
    for i in range(len(tri)):
        for j in range(i+1,len(tri)):
            r = shareEdge(tri[i],tri[j])
            if not r is None:   
                # plot_line(ax,r,w=1) 

                neighbours[i].append(j)
                neighbours[j].append(i)
    
    data = [ [i,(centerTriangle(tri[i])),neighbours[i] ] for i in range(len(tri))]

    g = Graph()
    g.createFromTriangle(data)
    
    paths =  g.findSimplePath(s,t)
    
    pathsToEnd = []
    for p in paths:
        line = [centerTriangle(tri[p[0]])]
        for i in range(1,len(p)-1):
            edge = getShareEdge(tri[p[i]],tri[p[i+1]])
            line += [midPoint(edge[0],edge[1]), centerTriangle(tri[p[i+1]])] 
        
        pathsToEnd.append(LineString(line))
    
    if not ax is None:
        for lineToPlot in pathsToEnd:
           plot_line(ax,lineToPlot,w=3,a = 0.4,c=numpy.random.rand(3,1))


    return pathsToEnd



def SendTriangle(func):
    triangles = []

    def inner(*args,**kargs):


        #Save the triangles
        if "triangles" in kargs:
            for t in list(kargs["triangles"]):            
                triangles.append(t)
            return 
        
        if len(triangles) > 0 and not isinstance(args[2],int):
            #Find the triangle for both
            i, j = 0,0
            start = Point(args[2])
            end = Point(args[3])
            for k in range(len(triangles)):
                t = triangles[k]
                if start.within( Polygon(arrayPoint(t))):
                    i = k
                if end.within( Polygon(arrayPoint(t))):
                    j = k
            args = list(args)
            args[2] = i
            args[3] = j
            return func(*args,**kargs)
        else:       
            return func(*args,**kargs)
    return inner


@SendTriangle
def getRegionPaths(multiLevelPoly,name,s,t,ax = None):
    
    pathsToEnd = getPathsFromTo(multiLevelPoly,name,s,t,ax=ax,visionCalculated=True)
    
    polyPathsVision = []
    data = list(open(name+"PolyPath.txt"))
    if len(data) == 0:

        polyPathsVision = getPolyVision(multiLevelPoly, pathsToEnd)
        data = open(name + "PolyPath.txt","w")
        for i in polyPathsVision:
            data.write(str(getPointsExport(i))+"\n" )
    else:
        for i in data:
            a = eval(i)

            if len(a)>0:    
                polyPathsVision.append(Polygon(a[0],a[1:]))
            else:
                polyPathsVision.append(Polygon(a[0]))

    return polyPathsVision


def getRegionPathsNoSave(multiLevelPoly,name,s,t,ax = None):
    
    pathsToEnd = getPathsFromTo(multiLevelPoly,name,s,t,ax=ax,visionCalculated=False)
    

    polyPathsVision = getPolyVisionOld(multiLevelPoly, pathsToEnd)

    return polyPathsVision


def GetPositionsContentShared(polyPathsVision,shared,ax=None):
    commonPosition = []

    
    pathPoly1 = polyPathsVision[0]


    
    intersectedPoly = polyPathsVision[0]
    
    for j in range(1,len(polyPathsVision)):
        
        pathPoly2 = polyPathsVision[j]

        intersectedPoly = intersectedPoly.intersection(pathPoly2)
    

    # pick the center to put the object. 
    sortedPlaces = []
    for a in intersectedPoly:
        if a.area>2:
            sortedPlaces.append((-a.area,a))
        
    sortedPlaces = sorted(sortedPlaces)


    if ax:
        for p in sortedPlaces:
            p = p[1]
            patch = PolygonPatch(p, facecolor="#FF00FF", edgecolor=v_color(p), alpha=0.1, zorder=2)
            ax.add_patch(patch)        


    for nbOb in range(0,shared):
        if sortedPlaces[nbOb][0] < - 10:
            commonPosition.append(sortedPlaces[nbOb][1].representative_point())        

    return commonPosition




######################    
######################    
######################    
######################    
######################





def SharedContent():

    name = "wasteland"
    # name = "leveltest"
    #Load the geometries
    multiLevelPoly = getLevelPolygon(name)

    outside = multiLevelPoly.bounds
    outside = Polygon([(outside[0],outside[1]),(outside[0],outside[3]),(outside[2],outside[3]),(outside[2],outside[1])])



    fig = pyplot.figure(1)
    ax = fig.add_subplot(111)

    
    #Set the triangles
    getRegionPaths(triangles=TriangulatePolygon(multiLevelPoly))
    
    polyPathsVision = getRegionPaths(multiLevelPoly,name,[50,350],[730,550])
    #This is the set up
    
    #Get the path 




    # Find the the common places
    
    shared = 2
    commonPosition = []

    
    pathPoly1 = polyPathsVision[0]

    for p in multiLevelPoly:
        patch = PolygonPatch(p, facecolor=v_color(p), edgecolor=v_color(p), alpha=0.5, zorder=2)
        ax.add_patch(patch)

    
    intersectedPoly = polyPathsVision[0]
    
    for j in range(1,len(polyPathsVision)):
        
        pathPoly2 = polyPathsVision[j]

        intersectedPoly = intersectedPoly.intersection(pathPoly2)
    

    # pick the center to put the object. 
    sortedPlaces = []
    for a in intersectedPoly:
        if a.area>2:
            sortedPlaces.append((-a.area,a))
        
    sortedPlaces = sorted(sortedPlaces)

    # for p in sortedPlaces:
    #     p = p[1]
    #     patch = PolygonPatch(p, facecolor="#FF00FF", edgecolor=v_color(p), alpha=0.1, zorder=2)
    #     ax.add_patch(patch)        


    for nbOb in range(0,shared):
        if sortedPlaces[nbOb][0] < - 10:
            commonPosition.append(sortedPlaces[nbOb][1].representative_point())        


    #####calculate the unique places here.





    plot_points(ax,commonPosition, c="#FFFF00")

    #Construct the graph data structure from the triangles. 
    #Add their id as welll 

    xrange = [outside.bounds[0], outside.bounds[2]]
    yrange = [outside.bounds[1], outside.bounds[3]]
    ax.set_xlim(*xrange)
    # ax.set_xticks(range(xrange[0],xrange[-1],100) )
    ax.set_ylim(*yrange)
    # ax.set_yticks(range(yrange[0],yrange[-1],100) )
    ax.set_aspect(1)
    ax.invert_yaxis()
    # pyplot.show()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    # s = [50,350]
    # t = [642,517]
    s = [50,350]
    t = [642,517]
    tris = TriangulatePolygon(multiLevelPoly)

    s_index = getTriangleIndex(TriangulatePolygon(multiLevelPoly),s)
    t_index = getTriangleIndex(TriangulatePolygon(multiLevelPoly),t)

    s = centerTriangle( arrayPoint( tris[s_index] ) )  
    t = centerTriangle( arrayPoint( tris[t_index] ) )  

    
    if name is "wasteland":
        a = [[798,115],[515,398],[829,407]]
        for i in a:
            plot_coords(ax,Point(i),c = "#ff0000")


        ax.text(s[0]+5,s[1]+30, r'$s$', fontsize=15)
        plot_coords(ax,Point(s),c = "#00FF00")

        ax.text(t[0]+10,t[1]+15, r'$t$', fontsize=15)
        plot_coords(ax,Point(t),c = "#FF0000",z=4)


    pyplot.savefig("Output_Shared_"+str(shared)+"_ob_"+str(name)+".pdf",bbox_inches='tight')   




def FindCommonPlacesNewInfrastructure():

    name = "wasteland"
    # name = "leveltest"
    #Load the geometries
    multiLevelPoly = getLevelPolygon(name)

    outside = multiLevelPoly.bounds
    outside = Polygon([(outside[0],outside[1]),(outside[0],outside[3]),(outside[2],outside[3]),(outside[2],outside[1])])



    fig = pyplot.figure(1)
    ax = fig.add_subplot(111)


    getRegionPaths(triangles=TriangulatePolygon(multiLevelPoly))
        
    polyPathsVision = getRegionPaths(multiLevelPoly,name,[50,350],[730,550])

    #This is the set up
    
    #Get the path 


    for p in multiLevelPoly:
        patch = PolygonPatch(p, facecolor=v_color(p), edgecolor=v_color(p), alpha=0.5, zorder=2)
        ax.add_patch(patch)

    # for p in polyPathsVision:
    #     patch = PolygonPatch(p, facecolor=v_color(p), edgecolor=v_color(p), alpha=0.1, zorder=2)
    #     ax.add_patch(patch)



    # Find the the common places
    
    shared = 2
    unique = 1
    ratio = 1

    commonPosition = GetPositionsContentShared(polyPathsVision,shared,ax=ax)




    plot_points(ax,commonPosition, c="#FFFF00")

    #Construct the graph data structure from the triangles. 
    #Add their id as welll 

    xrange = [outside.bounds[0], outside.bounds[2]]
    yrange = [outside.bounds[1], outside.bounds[3]]
    ax.set_xlim(*xrange)
    # ax.set_xticks(range(xrange[0],xrange[-1],100) )
    ax.set_ylim(*yrange)
    # ax.set_yticks(range(yrange[0],yrange[-1],100) )
    ax.set_aspect(1)
    ax.invert_yaxis()
    pyplot.show()





def DFSUniqueContent(unique, polys, toEval, depth):
    """This method finds the unique position with ratio
    """
    # print toEval
    
    # print depth
    random.shuffle(polys)
    if depth <=0:
        return toEval
    
    if toEval.area < 10:
        return None
    
    for j in range(0, len(polys)):
        
        pathPoly2 = polys[j]


        try:
            intersectedPoly = toEval.difference(pathPoly2)
            
            newArray = polys[0:j] + polys[j+1:]

            r = DFSUniqueContent(unique, newArray,intersectedPoly,depth-1)

            if not r is None :
                
                if r.geom_type is "Polygon" and r.area>10:
                    return r
                else:
                    sortedPlaces = []
                    
                    for a in intersectedPoly:
                        if a.area>10:
                            sortedPlaces.append((-a.area,a))
                
                        sortedPlaces = sorted(sortedPlaces)
                        
                        if len(sortedPlaces)>0:
                            return r
        except:
            # print pathPoly2
            pass               
    return toEval

def FindUniqueContent():

    name = "wasteland"
    # name = "leveltest"
    #Load the geometries
    multiLevelPoly = getLevelPolygon(name)

    outside = multiLevelPoly.bounds
    outside = Polygon([(outside[0],outside[1]),(outside[0],outside[3]),(outside[2],outside[3]),(outside[2],outside[1])])



    fig = pyplot.figure(1)
    ax = fig.add_subplot(111)


    getRegionPaths(triangles=TriangulatePolygon(multiLevelPoly))
        
    polyPathsVision = getRegionPaths(multiLevelPoly,name,[50,350],[730,550])



    paths = getPathsFromTo(multiLevelPoly,name,[50,350],[730,550])

    #This is the set up
    
    #Get the path 


    for p in multiLevelPoly:
        patch = PolygonPatch(p, facecolor=v_color(p), edgecolor=v_color(p), alpha=0.5, zorder=2)
        ax.add_patch(patch)




    # Find the the common places
    
    shared = 2
    unique = 1
    ratio = 0.5


    nbOptions = len(polyPathsVision)

    results = []

    # i = 2
    # results.append(DFSUniqueContent(unique,polyPathsVision[0:i] + polyPathsVision[i+1:],polyPathsVision[i],ratio*nbOptions))

    # plot_line(ax,paths[i])
    for i in range(len(polyPathsVision)):

        results.append(DFSUniqueContent(unique,polyPathsVision[0:i] + polyPathsVision[i+1:],polyPathsVision[i],ratio*nbOptions))    
    
    # polyPathsVision = list(reversed(polyPathsVision))
    # for i in range(len(polyPathsVision)):

    #     results.append(DFSUniqueContent(unique,polyPathsVision[0:i] + polyPathsVision[i+1:],polyPathsVision[i],ratio*nbOptions))    
    

    # print results
    uniquePositions = []
    for uniquePlacePolys in results:
        
    #     if uniquePlacePolys.geom_type is "Polygon":
    #         p = uniquePlacePolys
    #         patch = PolygonPatch(p, facecolor=v_color(p), edgecolor=v_color(p), alpha=0.1, zorder=2)
    #         ax.add_patch(patch)
    #     else:
    #         for p in uniquePlacePolys:
    #             if p.geom_type is "Polygon":
    #                 patch = PolygonPatch(p, facecolor=v_color(p), edgecolor=v_color(p), alpha=0.1, zorder=2)
    #                 ax.add_patch(patch)

        #only add if the distance is greater than some tresholds. 
        for v in uniquePositions:
            if LineString([v,uniquePlacePolys.representative_point()]).length < 10:
                break
        else:
            uniquePositions.append(uniquePlacePolys.representative_point())


    plot_points(ax,uniquePositions, c="#FF00FF")

    #Construct the graph data structure from the triangles. 
    #Add their id as welll 

    xrange = [outside.bounds[0], outside.bounds[2]]
    yrange = [outside.bounds[1], outside.bounds[3]]
    ax.set_xlim(*xrange)
    # ax.set_xticks(range(xrange[0],xrange[-1],100) )
    ax.set_ylim(*yrange)
    # ax.set_yticks(range(yrange[0],yrange[-1],100) )
    ax.set_aspect(1)

    ax.invert_yaxis()

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    # s = [50,350]
    # t = [642,517]
    s = [50,350]
    t = [642,517]
    tris = TriangulatePolygon(multiLevelPoly)

    s_index = getTriangleIndex(TriangulatePolygon(multiLevelPoly),s)
    t_index = getTriangleIndex(TriangulatePolygon(multiLevelPoly),t)

    s = centerTriangle( arrayPoint( tris[s_index] ) )  
    t = centerTriangle( arrayPoint( tris[t_index] ) )  

    
    if name is "wasteland":
        a = [[798,115],[515,398],[829,407]]
        for i in a:
            plot_coords(ax,Point(i),c = "#ff0000")


        ax.text(s[0]+5,s[1]+30, r'$s$', fontsize=15)
        plot_coords(ax,Point(s),c = "#00FF00")

        ax.text(t[0]+10,t[1]+15, r'$t$', fontsize=15)
        plot_coords(ax,Point(t),c = "#FF0000",z=4)


    pyplot.savefig("Output_Unique_"+str(unique)+"_r_"+str(ratio)+"_ob_"+str(name)+".pdf",bbox_inches='tight')   
    # pyplot.show()

def PlaceNeverSeen():
    name = "wasteland"
    # name = "leveltest"
    #Load the geometries
    multiLevelPoly = getLevelPolygon(name)

    outside = multiLevelPoly.bounds
    outside = Polygon([(outside[0],outside[1]),(outside[0],outside[3]),(outside[2],outside[3]),(outside[2],outside[1])])



    fig = pyplot.figure(1)
    ax = fig.add_subplot(111)


    getRegionPaths(triangles=TriangulatePolygon(multiLevelPoly))
    
    polyPathsVision = getRegionPaths(multiLevelPoly,name,[50,350],[642,517])

    visionAll = Polygon()
    for p in polyPathsVision:
        visionAll = visionAll.union(p)



    for p in multiLevelPoly:
        patch = PolygonPatch(p, facecolor=v_color(p), edgecolor=v_color(p), alpha=0.5, zorder=2)
        ax.add_patch(patch)




    p = visionAll
    # patch = PolygonPatch(p, facecolor=v_color(p), edgecolor=v_color(p), alpha=0.5, zorder=2)
    # ax.add_patch(patch)


    inside = outside.difference(multiLevelPoly) 

    r = inside.difference(visionAll)


    uniquePositions = []


    for p in r:
        # patch = PolygonPatch(p, facecolor="#FFAABB", edgecolor=v_color(p), alpha=0.5, zorder=2)
        # ax.add_patch(patch)
        if p.area > 100:
            uniquePositions.append(p.representative_point())                    



    plot_points(ax,uniquePositions, c="#00FFFF")

    #Construct the graph data structure from the triangles. 
    #Add their id as welll 

    xrange = [outside.bounds[0], outside.bounds[2]]
    yrange = [outside.bounds[1], outside.bounds[3]]
    ax.set_xlim(*xrange)
    # ax.set_xticks(range(xrange[0],xrange[-1],100) )
    ax.set_ylim(*yrange)
    # ax.set_yticks(range(yrange[0],yrange[-1],100) )
    ax.set_aspect(1)

    ax.invert_yaxis()

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    # s = [50,350]
    # t = [642,517]
    s = [50,350]
    t = [642,517]
    tris = TriangulatePolygon(multiLevelPoly)

    s_index = getTriangleIndex(TriangulatePolygon(multiLevelPoly),s)
    t_index = getTriangleIndex(TriangulatePolygon(multiLevelPoly),t)

    s = centerTriangle( arrayPoint( tris[s_index] ) )  
    t = centerTriangle( arrayPoint( tris[t_index] ) )  

    
    if name is "wasteland":
        a = [[798,115],[515,398],[829,407]]
        for i in a:
            plot_coords(ax,Point(i),c = "#ff0000")


        ax.text(s[0]+5,s[1]+30, r'$s$', fontsize=15)
        plot_coords(ax,Point(s),c = "#00FF00")

        ax.text(t[0]+10,t[1]+15, r'$t$', fontsize=15)
        plot_coords(ax,Point(t),c = "#FF0000",z=4)


    pyplot.savefig("Output_NeverSeen_"+str(name)+".pdf",bbox_inches='tight')   
    # pyplot.show()


def SavePathOnTopOfEachOtherWithVisionRegion():

    name = "wasteland"
    # name = "leveltest"
    #Load the geometries
    multiLevelPoly = getLevelPolygon(name)

    outside = multiLevelPoly.bounds
    outside = Polygon([(outside[0],outside[1]),(outside[0],outside[3]),(outside[2],outside[3]),(outside[2],outside[1])])



    fig = pyplot.figure(1)
    ax = fig.add_subplot(111)


    

    #This is the set up
    
    #Get the path 
    getRegionPaths(triangles=TriangulatePolygon(multiLevelPoly))    
    polyPathsVision = getRegionPaths(multiLevelPoly,name,[50,350],[730,550])
    
     

    

    for p in multiLevelPoly:
        patch = PolygonPatch(p, facecolor=v_color(p), edgecolor=v_color(p), alpha=0.5, zorder=2)
        ax.add_patch(patch)

    
    
    for i in polyPathsVision:
        
        patch = PolygonPatch(i, facecolor="#FF00FF", edgecolor=v_color(i), alpha=0.1, zorder=1)
        ax.add_patch(patch)
    




    #Construct the graph data structure from the triangles. 
    #Add their id as welll 

    xrange = [outside.bounds[0], outside.bounds[2]]
    yrange = [outside.bounds[1], outside.bounds[3]]
    ax.set_xlim(*xrange)
    # ax.set_xticks(range(xrange[0],xrange[-1],100) )
    ax.set_ylim(*yrange)
    # ax.set_yticks(range(yrange[0],yrange[-1],100) )
    ax.set_aspect(1)
    ax.invert_yaxis()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    pyplot.show()

def SaveAllVisibilityRegionPath():

    name = "wasteland"
    # name = "leveltest"
    #Load the geometries
    multiLevelPoly = getLevelPolygon(name)

    outside = multiLevelPoly.bounds
    outside = Polygon([(outside[0],outside[1]),(outside[0],outside[3]),(outside[2],outside[3]),(outside[2],outside[1])])



    fig = pyplot.figure(1)
    ax = fig.add_subplot(111)


    

    #This is the set up
    
    #Get the path 
    
    polyPathsVision = []
    data = list(open(name+"PolyPath.txt"))
    if len(data) == 0:
        polyPathsVision = getPolyVision(multiLevelPoly, pathsToEnd)
        data = open(name + "PolyPath.txt","w")
        for i in polyPathsVision:
            data.write(str(getPointsExport(i))+"\n" )
    else:
        for i in data:
            a = eval(i)

            if len(a)>0:    
                polyPathsVision.append(Polygon(a[0],a[1:]))
            else:
                polyPathsVision.append(Polygon(a[0]))

    
    



    for i in range(len(polyPathsVision)):
        # i = 75
        fig.clf()
        fig = pyplot.figure(1)
        ax = fig.add_subplot(111)

        pvision = polyPathsVision[i]
        # drawTriangle(ax,tri[i],id=i)



        for p in multiLevelPoly:
            patch = PolygonPatch(p, facecolor=v_color(p), edgecolor=v_color(p), alpha=0.5, zorder=2)
            ax.add_patch(patch)


        patch = PolygonPatch(pvision, facecolor="#FF00FF", edgecolor=v_color(pvision), alpha=0.1, zorder=1)
        ax.add_patch(patch)

        xrange = [outside.bounds[0], outside.bounds[2]]
        yrange = [outside.bounds[1], outside.bounds[3]]
        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)
        ax.set_aspect(1)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        
        ax.invert_yaxis()



        pyplot.savefig("output/polyPaths/vision_poly_"+str(i)+".pdf",bbox_inches='tight')   
    
    
def SaveAllVisibilityRegionPoints():
    name = "wasteland"
    # name = "leveltest"
    #Load the geometries
    multiLevelPoly = getLevelPolygon(name)

    outside = multiLevelPoly.bounds
    outside = Polygon([(outside[0],outside[1]),(outside[0],outside[3]),(outside[2],outside[3]),(outside[2],outside[1])])

    tri = TriangulatePolygon(multiLevelPoly)


    fig = pyplot.figure(1)
    ax = fig.add_subplot(111)


    

    GetVisionPolygon(triangle = tri)

    for i in range(len(tri)):
        # i = 75
        fig.clf()
        fig = pyplot.figure(1)
        ax = fig.add_subplot(111)

        # drawTriangle(ax,tri[i],id=i)
        point = centerTriangle(tri[i])
        plot_point(ax,point,c="#0ACC0A")


        for p in multiLevelPoly:
            patch = PolygonPatch(p, facecolor=v_color(p), edgecolor=v_color(p), alpha=0.5, zorder=2)
            ax.add_patch(patch)

        vision = GetVisionPolygon(multiLevelPoly,Point(point),ax)
        vision = GetVisionPolygon(multiLevelPoly,Point(point),ax)

        xrange = [outside.bounds[0], outside.bounds[2]]
        yrange = [outside.bounds[1], outside.bounds[3]]
        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)
        ax.set_aspect(1)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        
        ax.invert_yaxis()



        pyplot.savefig("output/vision_"+str(i)+".pdf",bbox_inches='tight')
        
        

def displayTriangulationRoadMap():
    name = "wasteland"
    # name = "leveltest"
    #Load the geometries
    multiLevelPoly = getLevelPolygon(name)

    outside = multiLevelPoly.bounds
    outside = Polygon([(outside[0],outside[1]),(outside[0],outside[3]),(outside[2],outside[3]),(outside[2],outside[1])])

    #This gives the definition of the interior polygon
    #From there we can do a triangulation
    #Hummmm
   
    

    #Draw triangulation
    fig = pyplot.figure(1)
    ax = fig.add_subplot(111)


    for p in multiLevelPoly:
       patch = PolygonPatch(p, facecolor=v_color(p), edgecolor=v_color(p), alpha=0.5, zorder=2)
       ax.add_patch(patch)


    tri = TriangulatePolygon(multiLevelPoly)
    for i in range(len(tri)):
        drawTriangle(ax,tri[i],id=i)
        plot_point(ax,centerTriangle(tri[i]))

    neighbours = [[] for i in range(len(tri))]
    for i in range(len(tri)):
        for j in range(i+1,len(tri)):
            r = shareEdge(tri[i],tri[j])
            if not r is None:   
                plot_line(ax,r,w=1) 


    neighbours = [[] for i in range(len(tri))]
    for i in range(len(tri)):
        for j in range(i+1,len(tri)):
            r = shareEdge(tri[i],tri[j])
            if not r is None:   
                # plot_line(ax,r,w=1) 

                neighbours[i].append(j)
                neighbours[j].append(i)
    
    data = [ [i,(centerTriangle(tri[i])),neighbours[i] ] for i in range(len(tri))]

    g = Graph()
    g.createFromTriangle(data)
    print g
    paths = g.findSimplePath(9,8)
    print paths

    #Construct the graph data structure from the triangles. 
    #Add their id as welll 

    xrange = [outside.bounds[0], outside.bounds[2]]
    yrange = [outside.bounds[1], outside.bounds[3]]
    ax.set_xlim(*xrange)
    # ax.set_xticks(range(xrange[0],xrange[-1],100) )
    ax.set_ylim(*yrange)
    # ax.set_yticks(range(yrange[0],yrange[-1],100) )
    ax.set_aspect(1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    # pyplot.show()
    ax.invert_yaxis()

    # pyplot.savefig("RoadMapTriangulation.pdf",bbox_inches='tight')
    pyplot.show()



def main():

    name = "level1"
    # name = "leveltest"
    #Load the geometries
    level = open(name+".txt")

    polyLevel = []


    for i in level: 
        p = eval(i)
        polyLevel.append(Polygon(p))

    multiLevelPoly = MultiPolygon(polyLevel)
    multiLevelPoly = multiLevelPoly.union(multiLevelPoly)

    outside = multiLevelPoly.bounds
    outside = Polygon([(outside[0],outside[1]),(outside[0],outside[3]),(outside[2],outside[3]),(outside[2],outside[1])])


    #LOAD the road map
    pathsToEnd = [] # This contains the lines from beginning to end. 
    # level = open(name+"r.txt")
    # for i in level:
    #     line = eval(i)
    #     pathsToEnd.append(LineString(line))

    # roadMap = MultiLineString(pathsToEnd)
    # pointsRoadMap = set()


    # for i in roadMap:
    #     for p in i.coords:
    #         pointsRoadMap.add(p)
        

    # pointsRoadMap = list(pointsRoadMap)
    pointsRoadMap = []
    g = Graph("level1graph.txt")
    for i in g.nodes:
        pointsRoadMap.append(i.pos)
    sol = g.findSimplePath(0,100)

    for s in sol:
        pathsToEnd.append(LineString([g.nodes[i].pos for i in s]))
    roadMap = MultiLineString (pathsToEnd)



    # ######## ##### ### ## ## ## ######


    ###DRAWING
    fig = pyplot.figure(1)
    ax = fig.add_subplot(111)


    for p in multiLevelPoly:
       patch = PolygonPatch(p, facecolor=v_color(p), edgecolor=v_color(p), alpha=0.5, zorder=2)
       ax.add_patch(patch)

    ###Drawing the road map
    plot_multiLines(ax,roadMap)

    plot_CoordsList(ax,pointsRoadMap)


    #####Drawing the fov
    # patch = PolygonPatch(fovPoly, facecolor="#A0F0AF", edgecolor=v_color(visionPoly), alpha=0.8, zorder=3)
    # ax.add_patch(patch)


    ####Single vision region: 
    p = pointsRoadMap[5]
    visionRegion = GetVisionPolygon(polyLevel,Point(p),ax)

    plot_coords(ax,Point(p),c = "#00FF00")




    ####Drawing the points for obj placement



    #####Draw the vision from a point
    lines = []
    # for i in vision:
    #     lines.append(i[0])
    # plot_multiLines(ax,vision, c = "#FF0000",a=0.5)



    ####Vision polygon

    # patch = PolygonPatch(polyPathsVision[0], facecolor="#FF00FF", edgecolor=v_color(polyPathsVision[0]), alpha=0.1, zorder=2)
    # ax.add_patch(patch)

    #### difference poly intersectedPoly

    #ploting
    xrange = [outside.bounds[0], outside.bounds[2]]
    yrange = [outside.bounds[1], outside.bounds[3]]
    ax.set_xlim(*xrange)
    # ax.set_xticks(range(xrange[0],xrange[-1],100) )
    ax.set_ylim(*yrange)
    # ax.set_yticks(range(yrange[0],yrange[-1],100) )
    ax.set_aspect(1)
    ax.invert_yaxis()
    pyplot.show()


def getLevelPolygon(name): 
    """ this new method works with only polygon
    """

    # name = "leveltest"
    #Load the geometries
    if not ".txt" in name:
        level = open(name+".txt")
    else:
        level = name
        
    polyLevel = []

    j = 0
    c = 0 
    for i in level: 
        p = eval(i)
        c+= len(p)
        if j == 0:
            p = Polygon(p)
            outside = p.bounds
            outside = Polygon([(outside[0]-10,outside[1]-10),(outside[0]-10,outside[3]+10),(outside[2]+10,outside[3]+10),(outside[2]+10,outside[1]-10)])
            polyLevel.append(outside.difference(p))

            j+=1
        else:    
            polyLevel.append(Polygon(p))

    multiLevelPoly = MultiPolygon(polyLevel)
    multiLevelPoly = multiLevelPoly.union(multiLevelPoly)

    return multiLevelPoly

def drawLevel():
    name = "wasteland"
    multiLevelPoly = getLevelPolygon(name)

    outside = multiLevelPoly.bounds
    outside = Polygon([(outside[0],outside[1]),(outside[0],outside[3]),(outside[2],outside[3]),(outside[2],outside[1])])


    #ploting
    fig = pyplot.figure(1)
    ax = fig.add_subplot(111)


    for p in multiLevelPoly:
        print p
        patch = PolygonPatch(p, facecolor=v_color(p), edgecolor=v_color(p), alpha=0.5, zorder=2)
        ax.add_patch(patch)
    xrange = [outside.bounds[0], outside.bounds[2]]
    yrange = [outside.bounds[1], outside.bounds[3]]
    ax.set_xlim(*xrange)
    # ax.set_xticks(range(xrange[0],xrange[-1],100) )
    ax.set_ylim(*yrange)
    # ax.set_yticks(range(yrange[0],yrange[-1],100) )
    ax.set_aspect(1)
    ax.invert_yaxis()
    pyplot.show()


    # pyplot.savefig("output.pdf")


import time
def testTimeWithSave():
    tStart = time.clock()
    name = "wasteland"
    # name = "leveltest"
    #Load the geometries
    multiLevelPoly = getLevelPolygon(name)



    outside = multiLevelPoly.bounds
    outside = Polygon([(outside[0],outside[1]),(outside[0],outside[3]),(outside[2],outside[3]),(outside[2],outside[1])])


    getRegionPaths(triangles=TriangulatePolygon(multiLevelPoly))

    # level3
    # s = [35, 30]
    # t = [320, 315]
    
    s = [50,350]
    t = [642,517]
    tris = TriangulatePolygon(multiLevelPoly)

    # getRegionPaths(triangles=TriangulatePolygon(multiLevelPoly))    
    s_index = getTriangleIndex(TriangulatePolygon(multiLevelPoly),s)
    t_index = getTriangleIndex(TriangulatePolygon(multiLevelPoly),t)


    polyPathsVision = getRegionPaths(multiLevelPoly,name,s_index,t_index)

    #This is the set up
    
    #Get the path 

    print time.clock() - tStart



    # Find the the common places
    
    shared = 2
    commonPosition = []

    
    pathPoly1 = polyPathsVision[0]


    intersectedPoly = polyPathsVision[0]
    
    for j in range(1,len(polyPathsVision)):
        
        pathPoly2 = polyPathsVision[j]

        intersectedPoly = intersectedPoly.intersection(pathPoly2)
    

    # pick the center to put the object. 
    sortedPlaces = []
    for a in intersectedPoly:
        if a.area>2:
            sortedPlaces.append((-a.area,a))
        
    sortedPlaces = sorted(sortedPlaces)


    for nbOb in range(0,shared):
        if sortedPlaces[nbOb][0] < - 10:
            commonPosition.append(sortedPlaces[nbOb][1].representative_point())        

    print time.clock() - tStart;

def testTimeNothingSave():
    tStart = time.clock()
    name = "wasteland"
    # name = "leveltest"
    #Load the geometries
    multiLevelPoly = getLevelPolygon(name)

    outside = multiLevelPoly.bounds
    outside = Polygon([(outside[0],outside[1]),(outside[0],outside[3]),(outside[2],outside[3]),(outside[2],outside[1])])


    getRegionPaths(triangles=TriangulatePolygon(multiLevelPoly))
    
    s = [35, 30]
    t = [320, 315]
    s = [50,350]
    t = [642,517]

    tris = TriangulatePolygon(multiLevelPoly)

    # getRegionPaths(triangles=TriangulatePolygon(multiLevelPoly))    
    s_index = getTriangleIndex(TriangulatePolygon(multiLevelPoly),s)
    t_index = getTriangleIndex(TriangulatePolygon(multiLevelPoly),t)


    polyPathsVision = getRegionPathsNoSave(multiLevelPoly,name,s_index,t_index)

    #This is the set up
    
    #Get the path 




    # Find the the common places
    
    shared = 2
    commonPosition = []

    
    print time.clock() - tStart


    pathPoly1 = polyPathsVision[0]


    intersectedPoly = polyPathsVision[0]
    
    for j in range(1,len(polyPathsVision)):
        
        pathPoly2 = polyPathsVision[j]

        intersectedPoly = intersectedPoly.intersection(pathPoly2)
    

    # pick the center to put the object. 
    sortedPlaces = []
    for a in intersectedPoly:
        if a.area>2:
            sortedPlaces.append((-a.area,a))
        
    sortedPlaces = sorted(sortedPlaces)


    for nbOb in range(0,shared):
        if sortedPlaces[nbOb][0] < - 10:
            commonPosition.append(sortedPlaces[nbOb][1].representative_point())        



    print time.clock() - tStart





# if __name__ == "__main__": mainTestGraph()
# if __name__ == "__main__": main()

# if __name__ == "__main__": drawLevel() #To display the roadMapOnly
# if __name__ == "__main__": displayTriangulationRoadMap() #To display the roadMapOnly

# if __name__ == "__main__": testTriangulationPolygon()
# if __name__ == "__main__": SaveAllVisibilityRegionPoints()

if __name__ == "__main__": SharedContent()
# if __name__ == "__main__": testTimeNothingSave()
# if __name__ == "__main__": testTimeWithSave()
# if __name__ == "__main__": FindCommonPlacesNewInfrastructure()
# if __name__ == "__main__": FindUniqueContent()
# if __name__ == "__main__": PlaceNeverSeen()

# if __name__ == "__main__": SavePathOnTopOfEachOtherWithVisionRegion()