from scipy.io import arff
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
import time

file_smile = "/Users/jonathan/SDBD/clustering-benchmark/src/main/resources/datasets/artificial/smile1.arff"
file_2d3c = "/Users/jonathan/SDBD/clustering-benchmark/src/main/resources/datasets/artificial/2d-3c-no123.arff"
file_2d4c = "/Users/jonathan/SDBD/clustering-benchmark/src/main/resources/datasets/artificial/2d-4c-no4.arff"
file_donut = "/Users/jonathan/SDBD/clustering-benchmark/src/main/resources/datasets/artificial/donut1.arff"
file_long1 = "/Users/jonathan/SDBD/clustering-benchmark/src/main/resources/datasets/artificial/long1.arff"
file_blobs = "/Users/jonathan/SDBD/clustering-benchmark/src/main/resources/datasets/artificial/blobs.arff"
file_smile2 = "/Users/jonathan/SDBD/clustering-benchmark/src/main/resources/datasets/artificial/smile2.arff"


def save_fig(x, y, labels, name):
    plt.figure()
    plt.scatter(x, y, c=labels, marker='x')
    plt.savefig(name)


def erase_file(file_path):
    file = open(file_path, "w")
    file.close()


def insert_section(file_path, section_name):
    f = open(file_path, "a")
    f.write("\n###### ###### " + section_name + " ###### ######\n\n")
    f.close()


def exo1():
    data_smile = arff.loadarff(open(file_smile, 'r'))
    smile = np.array(data_smile)[0]
    plt.scatter(smile['a0'], smile['a1'], c=smile['class'], marker='.')
    plt.show()

    data_2d3c = arff.loadarff(open(file_2d3c, 'r'))
    no123 = np.array(data_2d3c)[0]
    plt.scatter(no123['a0'], no123['a1'], c=no123['class'], marker='.')
    plt.show()

    data_donut = arff.loadarff(open(file_donut, 'r'))
    donut = np.array(data_donut)[0]
    plt.scatter(donut['a0'], donut['a1'], c=donut['class'], marker='.')
    plt.show()


def run_KMeans(nb_cluster, data, label):
    tmps1 = time.time()
    kmeans = KMeans(n_clusters=nb_cluster, init='k-means++')
    kmeans.fit(data)
    tmps2 = time.time() - tmps1
    coeff_silhou = metrics.silhouette_score(data, kmeans.labels_, metric='euclidean')  # doit être grand
    coeff_davies = davies_bouldin_score(data, kmeans.labels_)  # doit être petit
    f = open("./execution_time/kmeans_clustering/kmeans_clustering.txt", "a")
    msg_time = "Temps d'execution " + label + " = %f\n" % tmps2
    msg_silhou = "Coefficient de Silhouette = %f\n" % coeff_silhou
    msg_davies = "Coefficient de Davies = %f\n" % coeff_davies
    f.write(msg_time + msg_silhou + msg_davies)
    f.close()
    plt.scatter(nb_cluster, coeff_silhou, c='red', marker='.')
    plt.scatter(nb_cluster, coeff_davies, c='blue', marker='x')
    return kmeans


def runAndSave_KMeans(nb_cluster, data, x, y, name, name_fig):
    kmeans = run_KMeans(nb_cluster, data, "Kmeans- " + name + " [" + str(nb_cluster) + "]")
    save_fig(x, y, kmeans.labels_, "./kmeans_graph/" + name_fig)


def iter_KMeansClustering(data, name):
    insert_section("./execution_time/kmeans_clustering/kmeans_clustering.txt", "KMeans- " + name)
    plt.figure()
    for iter_cluster in range(2, 10):
        run_KMeans(iter_cluster, data, "Kmeans- " + name + " [" + str(iter_cluster) + "]")
    plt.savefig("./metrics/kmeans/" + name)


def Clustering_KMeans():
    erase_file("./execution_time/kmeans_clustering/kmeans_clustering.txt")
    insert_section("./execution_time/kmeans_clustering/kmeans_clustering.txt", "KMeans 2d-4c-no4"
                   + " nombre clusters fixé")

    data_2d4c = arff.loadarff(open(file_2d4c, 'r'))
    _2d4cno4 = np.array(data_2d4c, dtype=object)[0]
    _2d4cno4_train = list(zip(_2d4cno4['a0'], _2d4cno4['a1']))
    nb_cluster = 4

    save_fig(_2d4cno4['a0'], _2d4cno4['a1'], _2d4cno4['class'], "./kmeans_graph/2d-4c-no4")
    runAndSave_KMeans(nb_cluster, _2d4cno4_train, _2d4cno4['a0'], _2d4cno4['a1'], "2d-4c-no4", "2d-4c-no4_kmeans")
    iter_KMeansClustering(_2d4cno4_train, "2d-4c-no4")

    insert_section("./execution_time/kmeans_clustering/kmeans_clustering.txt", "KMeans long1"
                   + " nombre clusters fixé")

    data_long1 = arff.loadarff(open(file_long1, 'r'))
    _long1 = np.array(data_long1, dtype=object)[0]
    _long1_train = list(zip(_long1['a0'], _long1['a1']))
    nb_cluster = 2

    save_fig(_long1['a0'], _long1['a1'], _long1['class'], "./kmeans_graph/long1")
    runAndSave_KMeans(nb_cluster, _long1_train, _long1['a0'], _long1['a1'], "long1", "long1_kmeans")
    iter_KMeansClustering(_long1_train, "long1")

    insert_section("./execution_time/kmeans_clustering/kmeans_clustering.txt", "KMeans blobs"
                   + " nombre clusters fixé")

    data_blobs = arff.loadarff(open(file_blobs, 'r'))
    _blobs = np.array(data_blobs, dtype=object)[0]
    _blobs_train = list(zip(_blobs['x'], _blobs['y']))
    nb_cluster = 3

    save_fig(_blobs['x'], _blobs['y'], _blobs['class'], "./kmeans_graph/blobs")
    runAndSave_KMeans(nb_cluster, _blobs_train, _blobs['x'], _blobs['y'], "blobs", "blobs_kmeans")
    iter_KMeansClustering(_blobs_train, "blobs")

    insert_section("./execution_time/kmeans_clustering/kmeans_clustering.txt", "KMeans smile2"
                   + " nombre clusters fixé")

    smile2_data = arff.loadarff(open(file_smile2, 'r'))
    _smile2 = np.array(smile2_data, dtype=object)[0]
    smile2_train = list(zip(_smile2['a0'], _smile2['a1']))
    nb_cluster = 4

    save_fig(_smile2['a0'], _smile2['a1'], _smile2['class'], "./kmeans_graph/smile2")
    runAndSave_KMeans(nb_cluster, smile2_train, _smile2['a0'], _smile2['a1'], "smile2", "smile2_kmeans")
    iter_KMeansClustering(smile2_train, "smile2")


def run_AggloClustering(nb_cluster, data_train, linkage, label):
    tmps1 = time.time()
    agglo = AgglomerativeClustering(nb_cluster, linkage=linkage)
    agglo.fit(data_train)
    tmps2 = time.time() - tmps1
    coeff_silhou = metrics.silhouette_score(data_train, agglo.labels_, metric='euclidean')  # doit être grand
    coeff_davies = davies_bouldin_score(data_train, agglo.labels_)  # doit être petit
    f = open("./execution_time/agglo_clustering/agglo_clustering.txt", "a")
    msg_time = "Temps d'execution " + label + " = %f\n" % tmps2
    msg_silhou = "Coefficient de Silhouette = %f\n" % coeff_silhou
    msg_davies = "Coefficient de Davies = %f\n" % coeff_davies
    f.write(msg_time + msg_silhou + msg_davies)
    f.close()
    plt.scatter(nb_cluster, coeff_silhou, c='red', marker='.')
    plt.scatter(nb_cluster, coeff_davies, c='blue', marker='x')
    return agglo


def iter_AggloClustering(data, linkage, name):
    insert_section("./execution_time/agglo_clustering/agglo_clustering.txt", "Agglomeratif Clustering [" + name +
                   "] [" + linkage + "]")
    plt.figure()
    for iter_cluster in range(2, 10):
        run_AggloClustering(iter_cluster, data, linkage,
                            "Agglomeratif " + linkage + "- " + name + " [" + str(iter_cluster) + "] [" + linkage + "]")
    plt.savefig("./metrics/agglomeratif/" + name)


def runAndSave_Agglo(nb_cluster, data, linkage, name, x, y, name_fig):
    agglo = run_AggloClustering(nb_cluster, data, linkage,
                                "Agglomeratif " + linkage + "- " + name + " [" + str(
                                    nb_cluster) + "] [" + linkage + "]")
    save_fig(x, y, agglo.labels_, "./agglomeratif_graph/" + name_fig)


def Clustering_Agglomeratif():
    erase_file("./execution_time/agglo_clustering/agglo_clustering.txt")
    insert_section("./execution_time/agglo_clustering/agglo_clustering.txt", "Agglomeratif Clustering 2d-4c-no4"
                   + " nombre clusters fixé")

    data_2d4c = arff.loadarff(open(file_2d4c, 'r'))
    _2d4cno4 = np.array(data_2d4c, dtype=object)[0]
    _2d4cno4_train = list(zip(_2d4cno4['a0'], _2d4cno4['a1']))
    nb_cluster = 4

    runAndSave_Agglo(nb_cluster, _2d4cno4_train, "single", "2dc4", _2d4cno4['a0'], _2d4cno4['a1'], "2dc4_single")
    runAndSave_Agglo(nb_cluster, _2d4cno4_train, "average", "2dc4", _2d4cno4['a0'], _2d4cno4['a1'], "2dc4_average")
    runAndSave_Agglo(nb_cluster, _2d4cno4_train, "complete", "2dc4", _2d4cno4['a0'], _2d4cno4['a1'], "2dc4_complete")
    runAndSave_Agglo(nb_cluster, _2d4cno4_train, "ward", "2dc4", _2d4cno4['a0'], _2d4cno4['a1'], "2dc4_ward")

    iter_AggloClustering(_2d4cno4_train, "single", "2d4c")
    iter_AggloClustering(_2d4cno4_train, "average", "2d4c")
    iter_AggloClustering(_2d4cno4_train, "complete", "2d4c")
    iter_AggloClustering(_2d4cno4_train, "ward", "2d4c")

    insert_section("./execution_time/agglo_clustering/agglo_clustering.txt", "Agglomeratif Clustering blobs"
                   + " nombre clusters fixé")

    data_blobs = arff.loadarff(open(file_blobs, 'r'))
    _blobs = np.array(data_blobs, dtype=object)[0]
    _blobs_train = list(zip(_blobs['x'], _blobs['y']))
    nb_cluster = 3

    runAndSave_Agglo(nb_cluster, _blobs_train, "single", "blobs", _blobs['x'], _blobs['y'], "blobs_single")
    runAndSave_Agglo(nb_cluster, _blobs_train, "average", "blobs", _blobs['x'], _blobs['y'], "blobs_average")
    runAndSave_Agglo(nb_cluster, _blobs_train, "complete", "blobs", _blobs['x'], _blobs['y'], "blobs_complete")
    runAndSave_Agglo(nb_cluster, _blobs_train, "ward", "blobs", _blobs['x'], _blobs['y'], "blobs_ward")

    iter_AggloClustering(_blobs_train, "single", "blobs")
    iter_AggloClustering(_blobs_train, "average", "blobs")
    iter_AggloClustering(_blobs_train, "complete", "blobs")
    iter_AggloClustering(_blobs_train, "ward", "blobs")


def run_DBSCANClustering(distance, min_pts, data_train, label):
    tmps1 = time.time()
    dbscan = DBSCAN(eps=distance, min_sample=min_pts)
    dbscan.fit(data_train)
    tmps2 = time.time() - tmps1
    f = open("./execution_time/dbscan_clustering/dbscan_clustering.txt", "a")
    msg_time = "Temps d'execution " + label + " = %f\n" % tmps2
    f.write(msg_time)
    f.close()
    return dbscan


def runAndSave_DBSCAN(distance, min_pts, data, name, x, y, name_fig):
    dbscan = run_DBSCANClustering(distance, min_pts, data, "DBSCAN - " + name + " - dt[" + str(distance)
                                  + "] pts[" + str(min_pts) + "]")
    save_fig(x, y, dbscan.labels_, "./dbscan_graph/" + name_fig)


def Clustering_DBSCAN():
    erase_file("./execution_time/dbscan_clustering/dbscan_clustering.txt")
    insert_section("./execution_time/dbscan_clustering/dbscan_clustering.txt", "DBSCAN 2d-4c-no4"
                   + " distance et nombre de points fixés")

    data_2d4c = arff.loadarff(open(file_2d4c, 'r'))
    _2d4cno4 = np.array(data_2d4c, dtype=object)[0]
    _2d4cno4_train = list(zip(_2d4cno4['a0'], _2d4cno4['a1']))
    distance = 5
    min_pts = 0.5

    runAndSave_DBSCAN(distance, min_pts, _2d4cno4_train, "2dc4", _2d4cno4['a0'], _2d4cno4['a1'], "2dc4")


def main():
    # exo1()
    # Clustering_KMeans()
    # Clustering_Agglomeratif()
    Clustering_DBSCAN()


if __name__ == "__main__":
    main()
