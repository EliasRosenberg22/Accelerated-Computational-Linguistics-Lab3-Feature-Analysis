#Author: Elias Rosenberg
#Date: April 19th, 2021
#Purpose: Creates a KMeans matrix from shakespeare plays and titles, creating a cluster of play titles by their related words.
#Then, convert that matrix into a visual dendrogram.
#Inputs: Shakespeare.txt (scripts) and shakespearePlayTitles.txt (titles)
#outputs: output window shows the clustering of plays by their words. Dendrogram code should produce a dendrogram pdf with the play titles.


#=========================================================================
# Dartmouth College, LING48, Spring 2021
# Rolando Coto-Solano (Rolando.A.Coto.Solano@dartmouth.edu)
# Examples for Week 03.2: Clustering Shakespeare
#
# Based on code from:
# https://pythonprogramminglanguage.com/kmeans-text-clustering/
#=========================================================================

# Read files and split the lines
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
# Libraries for dendrogram
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from numpy import ndarray
import numpy


#dendrogram stuff


titles = 'shakespearePlayTitles.txt'
allText = 'shakespeare.txt'
textBoundary = '<<NEWTEXTSTARTSHERE>>'

playTitles = open(titles, "r", encoding="utf8").read()
playTitles = playTitles.split("\n")

playScripts = open(allText, "r", encoding="utf8").read()
playScripts = playScripts.split(textBoundary)


Scripts = [] #getting the scripts out of the text file
for line in playScripts:
    Scripts.append(line)
Titles = [] #getting the titles out of the text file
for name in playTitles:
    Titles.append(name)
#print(Titles)
documents = Scripts

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents) #turning the scripts into a TF-IDF matrix

true_k = 10 #number of clusters suggested
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

names = model.labels_ #getting the lables for each cluster

newTitles = []
for idx in names:
    newTitles.append(Titles[idx]) #finding the titles at each given cluster index
#print(names)
#print(newTitles)

print("Plays and their word clusters")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print(newTitles[i] + " : cluster " + str(i)),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print()
print("\n")
print("Prediction")

Y = vectorizer.transform(["battle and king"])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["wit and love"])
prediction = model.predict(Y)
print(prediction)
#______________________________________________________________________________________________________________

#code for creating the dendrogram

# Import the dataset
new = X.todense() #converting the dataset to the right format
lables = numpy.array(Titles) #getting the proper format for lables
Z = linkage(new, 'ward') # Calculate the distance between each sample using 'ward' distancing

# Make the dendrogram
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance (ward)')
dendrogram(Z, labels=lables, orientation="left", truncate_mode='level', leaf_rotation=0, leaf_font_size=6)
plt.savefig('ie-dendrogram.pdf')
