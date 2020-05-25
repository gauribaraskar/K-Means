import random
import numpy as np
import math
from collections import defaultdict
import plotly
from plotly.graph_objs import Scatter, Scatter3d, Layout

# This is a python implementation of the K-Means/K-medians algorithm from scratch. The algorithm supports three distance Euclidean,Manhattan, Minkowski distances. 
#  
'''Helper fucntions'''
'''Function to return a Euclidean distance between 2 N-dimensional points'''
def getDistanceEuclidean(a,b):
	if(a.dim!=b[0].dim):
		raise Exception("Size of array is not same!")
	distances = []
	for j in range(len(b)):
		accumulatedDifference = 0.0
		for i in range(a.dim):
			squareDifference = pow((a.coords[i]-b[j].coords[i]), 2)
			accumulatedDifference += squareDifference
		distances.append(pow(accumulatedDifference,1/2))
	#print(distances)
	#print(distances.index(min(distances)))
	return distances.index(min(distances))

''' Fucntion to calculate Manhattan distance between 2 N-Dimensional point'''
def getDistanceManhattan(a,b):
	if(a.dim!=b[0].dim):
		raise Exception("Size of array is not same!")
	distances = []
	for j in range(len(b)):
		accumulatedDifference = 0.0
		for i in range(a.dim):
			squareDifference = abs(a.coords[i]-b[j].coords[i])
			accumulatedDifference += squareDifference
		distances.append(accumulatedDifference)
	return distances.index(min(distances))

def getDistanceMinkowski(a,b):
	if(a.dim!=b[0].dim):
		raise Exception("Size of array is not same!")
	distances = []
	for j in range(len(b)):
		accumulatedDifference = 0.0
		for i in range(a.dim):
			squareDifference = pow((a.coords[i]-b[j].coords[i]),a.dim)
			accumulatedDifference += squareDifference
		distances.append(pow((accumulatedDifference),float(1/a.dim)))
	return distances.index(min(distances))

'''Function to generate initial random seeds for clusters'''
def generateCentroids(data,num_clusters):
	list_of_centers = random.sample(range(len(data)-1), num_clusters)
	return list_of_centers

'''Function for assigning cluster to each data point based on the distance metric chosen'''
def assignCluster(data,init_centroids,distance):
	cluster_list = []
	#print("LENGTH",len(init_centroids))
	if(distance=='Euclidean'):
		for i in range(len(data)):
			#print(type(data[i]),type(init_centroids[0]))
			cluster_list.append(getDistanceEuclidean(data[i],init_centroids))
	elif(distance=='Manhattan'):
		for i in range(len(data)):
			cluster_list.append(getDistanceManhattan(data[i],init_centroids))
	elif(distance=='Minkowski'):
		for i in range(len(data)):
			cluster_list.append(getDistanceMinkowski(data[i],init_centroids))
	return cluster_list

'''Function to update centroids based on the method specified'''
def updateCentroids(use_means,data,listOfAssignedClusters):
	clusterIndices = defaultdict(list)
	rowOfData = 0
	for i in listOfAssignedClusters:
		clusterIndices[i].append(rowOfData)
		rowOfData += 1
	newCentroids = []
	if(use_means):
		for i in clusterIndices.keys():
			newCentroids.append([i,np.mean([np.array(data[m].coords) for m in clusterIndices[i]],axis=0)])
		newCentroids.sort(key=lambda x: x[0])
	else:
		#print("Using median")
		for i in clusterIndices.keys():
			newCentroids.append([i,np.median([np.array(data[m].coords) for m in clusterIndices[i]],axis=0)])
		newCentroids.sort(key=lambda x: x[0])
	return([Point(newCentroids[m][1]) for m in range(len(newCentroids))])

'''Classes'''
'''Class Point which stores each datapoint of n dimension'''
class Point:
	def __init__(self,coords):
		self.coords = coords
		self.dim = len(coords)

'''Class Kmeans to generate an object for this clustering'''
class KMeans:
	def __init__(self,num_clusters,num_iterations=10,init_centroids=None,distance=None,use_means=True):
		self.num_clusters = int(num_clusters)
		self.num_iterations = int(num_iterations)
		self.init_centroids = init_centroids
		self.distance = distance
		self.use_means = use_means
		if(distance) is None:
			self.distance = 'Euclidean'

	def fit(self,data):
		if self.init_centroids is None:
			list_centroids = generateCentroids(data,self.num_clusters)
			self.init_centroids = [data[i] for i in list_centroids]
		num_iter = 0
		updatedCentroids = self.init_centroids
		while(num_iter<self.num_iterations):
			listOfAssignedClusters = assignCluster(data,updatedCentroids,self.distance)
			updatedCentroids =  updateCentroids(self.use_means,data,listOfAssignedClusters)
			num_iter += 1
		return listOfAssignedClusters,updatedCentroids

class Cluster(object):
    def __init__(self, points):
        '''
        points - A list of point objects
        '''
        if len(points) == 0:
            raise Exception("ERROR: empty cluster")

        # The points that belong to this cluster
        self.points = points

        # The dimensionality of the points in this cluster
        self.n = points[0].dim
        
        # Assert that all points are of the same dimensionality
        for p in points:
            if p.dim != self.n:
                raise Exception("ERROR: inconsistent dimensions")

        # Set up the initial centroid (this is usually based off one point)
        self.centroid = self.calculateCentroid()

    def __repr__(self):
        '''
        String representation of this object
        '''
        return str(self.points)

    def update(self, points):
        '''
        Returns the distance between the previous centroid and the new after
        recalculating and storing the new centroid.
        Note: Initially we expect centroids to shift around a lot and then
        gradually settle down.
        '''
        old_centroid = self.centroid
        self.points = points
        # Return early if we have no points, this cluster will get
        # cleaned up (removed) in the outer loop.
        if len(self.points) == 0:
            return 0

        self.centroid = self.calculateCentroid()
        shift = getDistance(old_centroid, self.centroid,self.distanceMetric)
        return shift

    def calculateCentroid(self):
        '''
        Finds a virtual center point for a group of n-dimensional points
        '''
        numPoints = len(self.points)
        # Get a list of all coordinates in this cluster
        coords = [p.coords for p in self.points]
        # Reformat that so all x's are together, all y'z etc.
        unzipped = zip(*coords)
        # Calculate the mean for each dimension
        centroid_coords = [math.fsum(dList)/numPoints for dList in unzipped]

        return Point(centroid_coords)

    def getTotalDistance(self):
        '''
        Return the sum of all squared Euclidean distances between each point in 
        the cluster and the cluster's centroid.
        '''
        sumOfDistances = 0.0
        for p in self.points:
            sumOfDistances += getDistance(p, self.centroid,self.distanceMetric)

        return sumOfDistances

'''Helper functions to generate data'''
def generatePoint(dimension_of_point):
		p = Point([random.uniform(0,200) for _ in range(int(dimension_of_point))])
		return p

def generate_data(number_of_points,dimension_of_point):
	points = []
	for i in range(int(number_of_points)):
		points.append(generatePoint(dimension_of_point))
	return points
def plotClusters(data, dimensions):

    '''
    This uses the plotly offline mode to create a local HTML file.
    This should open your default web browser.
    '''
    if dimensions not in [2, 3]:
        raise Exception("Plots are only available for 2 and 3 dimensional data")

    # Convert data into plotly format.
    traceList = []
    for i, c in enumerate(data):
        # Get a list of x,y coordinates for the points in this cluster.
        cluster_data = []
        for point in c.points:
            cluster_data.append(point.coords)

        trace = {}
        centroid = {}
        if dimensions == 2:
            # Convert our list of x,y's into an x list and a y list.
            trace['x'], trace['y'] = zip(*cluster_data)
            trace['mode'] = 'markers'
            trace['marker'] = {}
            trace['marker']['symbol'] = i
            trace['marker']['size'] = 12
            trace['name'] = "Cluster " + str(i)
            traceList.append(Scatter(**trace))
            # Centroid (A trace of length 1)
            centroid['x'] = [c.centroid.coords[0]]
            centroid['y'] = [c.centroid.coords[1]]
            centroid['mode'] = 'markers'
            centroid['marker'] = {}
            centroid['marker']['symbol'] = i
            centroid['marker']['color'] = 'rgb(200,10,10)'
            centroid['name'] = "Centroid " + str(i)
            traceList.append(Scatter(**centroid))
        else:
            symbols = [
                "circle",
                "square",
                "diamond",
                "circle-open",
                "square-open",
                "diamond-open",
                "cross", "x"
            ]
            symbol_count = len(symbols)
            if i > symbol_count:
                print("Warning: Not enough marker symbols to go around")
            # Convert our list of x,y,z's separate lists.
            trace['x'], trace['y'], trace['z'] = zip(*cluster_data)
            trace['mode'] = 'markers'
            trace['marker'] = {}
            trace['marker']['symbol'] = symbols[i]
            trace['marker']['size'] = 12
            trace['name'] = "Cluster " + str(i)
            traceList.append(Scatter3d(**trace))
            # Centroid (A trace of length 1)
            centroid['x'] = [c.centroid.coords[0]]
            centroid['y'] = [c.centroid.coords[1]]
            centroid['z'] = [c.centroid.coords[2]]
            centroid['mode'] = 'markers'
            centroid['marker'] = {}
            centroid['marker']['symbol'] = symbols[i]
            centroid['marker']['color'] = 'rgb(200,10,10)'
            centroid['name'] = "Centroid " + str(i)
            traceList.append(Scatter3d(**centroid))

    title = "K-means clustering with %s clusters" % str(len(data))
    plotly.offline.plot({
        "data": traceList,
        "layout": Layout(title=title)
    })
'''main function to take user input'''
def main():
	distance = ['Euclidean','Manhattan','Minkowski']
	number_of_points = input("Enter the number of points: ")
	number_of_clusters = input("Enter the number of clusters: ")
	dimension_of_point = input("Enter the dimension of each data point: ")
	number_of_iterations = input("Enter the number of iterations you would like: ")
	distance_metric = input("Enter the distance metric you would like to use:\n1.Euclidean\n2.Manhattan\n3.Minkowski:\n(1/2/3?): ")
	distance_metric = distance[int(distance_metric)-1]
	use_means = input("Would you like to use K-Medians(Y/N)?: ")
	if(use_means=="Y" or use_means=="y"):
		use_means= True
	else:
		use_means = False
	if(int(number_of_clusters)>int(number_of_points)):
		raise Exception("Number of clusters cannot be greater than the number of points!")
	if(int(number_of_clusters)==1):
		raise Exception("Number of clusters should be greater than one!")
	data = generate_data(number_of_points,dimension_of_point)
	data = np.array(data)

	#print([data[m].coords for m in range(len(data))])
	kmeans = KMeans(number_of_clusters,number_of_iterations,distance=distance_metric,use_means=use_means)
	listOfFinalClusters,listOfFinalCentroids = kmeans.fit(data)
	print(listOfFinalClusters)
	clusters = []
	clusterIndices = defaultdict(list)
	rowOfData = 0
	for i in listOfFinalClusters:
		clusterIndices[i].append(data[rowOfData])
		rowOfData+1
	for key in clusterIndices.keys():
		clusters.append(Cluster(clusterIndices[key]))
	plotClusters(clusters,int(dimension_of_point))



if __name__== "__main__":
	main()
