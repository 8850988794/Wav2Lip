import csv
import os
import pandas as pd
import math
import numpy as np
import random
from collections import defaultdict
import networkx as nx

rows = csv.reader(open('path'))

node_set = set()
in_neighbour = defaultdict(set)
out_neighbour = defaultdict(set)
for instance in rows:
    source,sink_list = instance[0],instance[1:]
    node_set.add(source)
    out_neighbour[source] = set(sink_list)
    for node in sink_list:
        node_set.add(node)
        in_neighbour[node].add(source)
#print (in_neighbour)
print (len(node_set))

rows = csv.reader(open('path'))

g={}
for row in rows:
	g[row[0]] = row[1:]

clean_g={}
for key in g:
    if len(g[key]) > 0:
        clean_g[key] = g[key]

G = nx.DiGraph(g)
community = nx.community.louvain_communities(G,seed=123,resolution=1.0)
community_list = list(community)
non_edges=list(nx.non_edges(G))

def resource_allocation(node1,node2):
    result = 0
    common1 = in_neighbour[node1].intersection(in_neighbour[node2]).union(out_neighbour[node1].intersection(out_neighbour[node2]))
    common2 = in_neighbour[node1].intersection(out_neighbour[node2]).union(out_neighbour[node1].intersection(in_neighbour[node2]))
    common = common1.union(common2)
    for node in common:
        result += 1/(len(out_neighbour[node])+len(in_neighbour[node]))
    return result

def shared(node1,node2):
    return len(in_neighbour[node1].intersection(in_neighbour[node2])) + len(out_neighbour[node1].intersection(out_neighbour[node2]))

def knn(node1,node2):
    result=[]
    sink_in =1/math.sqrt(1+len(in_neighbour[node2]))
    source_out=1/math.sqrt(1+len(out_neighbour[node1]))
    result.append(source_out)
    result.append(sink_in)
    result.append(sink_in+source_out)
    result.append(sink_in*source_out)

    return result

# |A∩B|/|A∪B|

def transitive_friends(node1, node2):
  return len(out_neighbour[node1].intersection(in_neighbour[node2]))


def jaccard_distance(node1, node2):
    in_neighbour_union = len(in_neighbour[node1].union(in_neighbour[node2]))
    out_neighbour_union = len(out_neighbour[node1].union(out_neighbour[node2]))

    if in_neighbour_union == 0 and out_neighbour_union == 0:
        return 0

    return shared(node1,node2)/(in_neighbour_union+out_neighbour_union)

def cosine_distance(node1, node2):
    source_degrees = len(in_neighbour[node1]) + len(out_neighbour[node1])
    sink_degrees = len(in_neighbour[node2]) + len(out_neighbour[node2])

    if source_degrees == 0 or sink_degrees == 0:
        return 0

    return shared(node1, node2)/math.sqrt(source_degrees*sink_degrees)

#Sxy=∑z∈Γ(x)∩Γ(y) 1/logkz.
def adamic_adar(node1, node2):
    result = 0

    intersection = in_neighbour[node1].intersection(in_neighbour[node2]).union(out_neighbour[node1].intersection(out_neighbour[node2]))
    for node in intersection:
        degree = len(in_neighbour[node]) + len(out_neighbour[node])
        if degree == 1:
            continue
        result += 1/math.log(degree)
    return result

# Sxy=kx⋅ky
def preferential_attachment(node1, node2):
    source_degree = len(in_neighbour[node1])+len(out_neighbour[node1])
    sink_degree = len(in_neighbour[node2])+len(out_neighbour[node2])
    return source_degree*sink_degree


def follow_back(node1, node2):
    try:
        if node2 in in_neighbour[node1]:
            return int(1)
        else:
            return int(0)
    except:
        return int(0)

def friends_measure(node1,node2):
    result = 0
    node1_neighbours = in_neighbour[node1].union(out_neighbour[node1])
    #print(len(node1_neighbours))
    node2_neighbours = in_neighbour[node2].union(out_neighbour[node2])
    for node1 in node1_neighbours:
      node1_n = in_neighbour[node1].union(out_neighbour[node1])
      if len(node1_n.intersection(node2_neighbours)):
            result+=1
    return result


def cal_follows(node1, node2):
    numFollowersSource = len(in_neighbour[node1])
    numFolloweesSource = len(out_neighbour[node1])
    numFollowersSink = len(in_neighbour[node2])
    numFolloweesSink = len(out_neighbour[node2])
    numCommonFollowers = len(in_neighbour[node1].intersection(in_neighbour[node2]))
    numCommonFollowees = len(out_neighbour[node1].intersection(out_neighbour[node2]))

    return [numFollowersSource, numFolloweesSource, numFollowersSink, numFolloweesSink, numCommonFollowers, numCommonFollowees]

def katz_measure(node1, node2, beta=0.5, max_iterations = 10):
    result = 0
    node1_neighbours = in_neighbour[node1].union(out_neighbour[node1])
    node2_neighbours = in_neighbour[node2].union(out_neighbour[node2])

    for _ in range(max_iterations):
        for node in node1_neighbours:
            if node in node2_neighbours:
                result += beta ** _
        for node in node2_neighbours:
            if node in node1_neighbours:
                result += beta ** _

    return result

def bayesian_sets(node1, node2):
    node1_neighbors = in_neighbour[node1].union(out_neighbour[node1])
    node2_neighbors = in_neighbour[node2].union(out_neighbour[node2])
    common_neighbors = node1_neighbors.intersection(node2_neighbors)
    bayesian_sets_score = len(common_neighbors) / (len(node1_neighbors) + len(node2_neighbors) - len(common_neighbors))

    return bayesian_sets_score

def power_law_exponent(node1, node2):
    node1_neighbours = in_neighbour[node1].union(out_neighbour[node1])
    node2_neighbours = in_neighbour[node2].union(out_neighbour[node2])
    node1_degree = len(node1_neighbours)
    node2_degree = len(node2_neighbours)
    common_neighbors = node1_neighbours.intersection(node2_neighbours)
    num_common_neighbors = len(common_neighbors)
    if num_common_neighbors == 0:
        return 0
    exponent = (num_common_neighbors ** 2) / (node1_degree * node2_degree)

    return exponent

def same_community(node1, node2):
  for community in community_list:
    if node1 in community and node2 in community:
      return 1
  return 0

def get_feature(node1,node2):
    result=[]
    result.append(resource_allocation(node1,node2))
    result.append(shared(node1,node2))
    result.extend(knn(node1,node2))
    result.append(jaccard_distance(node1,node2))
    result.append(cosine_distance(node1,node2))
    result.append(adamic_adar(node1,node2))
    result.append(preferential_attachment(node1,node2))
    result.append(follow_back(node1,node2))
    result.extend(cal_follows(node1, node2))
    result.append(friends_measure(node1,node2))
    result.append(transitive_friends(node1,node2))
    result.append(same_community(node1,node2))
   

    return result


COLUMNS = [
    "resource_allocation",
    "shared",
    "k1",
    "k2",
    "k3",
    "k4",
    "jaccard_distance",
    "cosine_distance",
    "adamic_adar",
    "preferential_attachment",
    "follow_back",
    "numfollowerssource",
    "numfolloweessource",
    "numfollowerssink",
    "numfolloweessink",
    "numcommonfollowers",
    "numcommonfollowees",
    "friends_measure",
    "transitive_friends",
    "same_community"
]



def get_trainset(n_neg,n_pos):

    training_set = []

    for i in range(n_neg):
        if i%100==0:
            # clear_output(wait=True)
            # display('===>' + str(i) +  '/' + str(n_neg) )
            pass

        # source_array=np.array(list(clean_g.keys()))
        # sink_choice=np.array(list(clean_g.keys()))

        # sink_source=np.random.choice(sink_choice)

        # source=np.random.choice(source_array)
        # sink=np.random.choice(np.array(clean_g[sink_source]))

        rand_edges=np.random.choice(np.array(non_edges))
        source=rand_edges[0]
        sink=rand_edges[1]
        #print(source)
        #source=str(source)
        #print(source)
        #sink=str(sink)

        if sink in clean_g[source]:
            label=1
        else:
            label=0

        outcome = get_feature(source,sink)

        outcome.append(label)

        training_set.append(outcome)

    #pos
    for i in range(n_pos):
        if i%100==0:
            # clear_output(wait=True)
            # display('===>' + str(i) +  '/' + str(n_neg) )
            pass

        # source_list=np.array(list(clean_g.keys()))
        rand_edges=np.random.choice(np.array(non_edges))
        source=rand_edges[0]
        sink=rand_edges[1]

        #source=str(source)
        #sink=str(sink)

        if sink in clean_g[source]:
            label=1
        else:
            label=0

        outcome = get_feature(source,sink)

        outcome.append(label)

        training_set.append(outcome)

    print(len(training_set))
    print(training_set[0])


    columns= COLUMNS + ['label']
    try:
      data=pd.DataFrame(training_set,columns=columns)
    except:
      return training_set

    return data


if __name__ =='__main__':

    data = get_trainset(10000, 10000)

    try:
        pd.DataFrame(data, COLUMNS).to_csv('training_features_df.csv')
    except:
        with open('training_features.csv' , 'w') as f:
            w = csv.writer(f)
            for d in data:
                w.writerow(d)
    

