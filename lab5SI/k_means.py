import random

import numpy as np


def initialize_centroids_forgy(data, k):
    totalCandidates = data.shape[0];
    randomCandidates = np.random.choice(totalCandidates, size=k, replace=False)
    result = data[randomCandidates]
    return result


def initialize_centroids_kmeans_pp(data, k):
    totalCandidates = data.shape[0];
    firstCandidate = np.random.choice(totalCandidates)
    result = []
    result.append(data[firstCandidate])

    while len(result) < k:
        biggestDiff = -1
        winner = None
        for candidate in data:
            smallestDiff = np.inf
            for currentCentro in result:
                dist = np.linalg.norm(currentCentro - candidate)
                if dist < smallestDiff:
                    smallestDiff = dist
            if smallestDiff > biggestDiff:
                winner = candidate
                biggestDiff = smallestDiff
        result.append(winner)
    return np.array(result)


def assign_to_cluster(data, centroid):
    result = []
    for i in data:
        smallestDiff = np.inf
        currentCentro = -1
        for j, c in enumerate(centroid):
            dist = np.linalg.norm(i - c)
            if dist < smallestDiff:
                smallestDiff = dist
                currentCentro = j
        result.append(currentCentro)
    return np.array(result)


def update_centroids(data, assignments):
    new_centroids = []

    for cluster_id in np.unique(assignments):
        cluster_points = data[assignments == cluster_id]
        new_center = np.mean(cluster_points, axis=0)
        new_centroids.append(new_center)

    return np.array(new_centroids)


def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :]) ** 2))


def k_means(data, num_centroids, kmeansplusplus=False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else:
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments = assign_to_cluster(data, centroids)
    for i in range(100):  # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments):  # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)
