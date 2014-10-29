#!/usr/bin/env python
#encoding=utf8

#Copyright [2014] [Wei Zhang]

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.


###################################################################
# Date: 2014/10/21                                                #
# Compare different geo-based methods for event recommendation.   #
# Implement follow models:                                        #
#   1.GeoMF (DPGMM+Gaussian Prob/Kernel method)                   #
#   2.GeoMF (KMeans+Gaussian Prob/Kernel method)                  #
#   3.GeoMF-O (Original, KDD14)                                   #
#   4.HeSig (AAAI14)                                              #
#   5.Distance Model (SIGIR11, KDD13, ...)                        #
###################################################################


import numpy as np
import json, csv, sys, random, pickle
sys.path.append("../../")
sys.path.append("../../../lib/geopy-1.3.0/")
from sklearn.mixture import DPGMM, GMM
from sklearn.cluster import KMeans
from geopy.distance import GreatCircleDistance
from collections import defaultdict
from tool import rZero, rPosGaussian, rGaussian, tic, toc

settings = json.loads(open("../../../SETTINGS.json").read())

MIN_EPSILON = 1e-10
#DIS_STD = 0.33
DIS_STD = 0.66

def checkGeoScope(poi, city_id):
    if poi[0]>settings["GEO_SCOPE"][city_id][0][0] and poi[0]<settings["GEO_SCOPE"][city_id][1][0]\
            and poi[1]>settings["GEO_SCOPE"][city_id][0][1] and poi[1]<settings["GEO_SCOPE"][city_id][1][1]:
        return True
    return False


def removeDup(dup_centers):
    clean_centers = []
    pool = set([])
    for center in dup_centers:
        key = str(center[0])+str(center[1])
        if key not in pool:
            pool.add(key)
            clean_centers.append(center)
    return clean_centers


def calCenterCov(n_cluster, labels, pois):
    labeled_pois = [[] for i in xrange(n_cluster)]
    for i, label in enumerate(labels):
        labeled_pois[label].append(pois[i])
    means = []
    for i in xrange(n_cluster):
        means.append(np.mean(labeled_pois[i], 0))
    variances = []
    for i in xrange(n_cluster):
        variances.append(np.sum((np.array(labeled_pois[i])-means[i])**2, 0)/(len(labeled_pois[i])))
    return means, variances


def calDisVariance(n_cluster, labels, pois):
    labeled_pois = [[] for i in xrange(n_cluster)]
    for i, label in enumerate(labels):
        labeled_pois[label].append(pois[i])
    means = []
    for i in xrange(n_cluster):
        means.append(np.mean(labeled_pois[i], 0))
    dis_variances = []
    for i in xrange(n_cluster):
        dis = np.sqrt(np.sum((np.array(labeled_pois[i])-means[i])**2,1))
        dis_variances.append(np.var(dis))
    return dis_variances


def calDistanceStd(n_cluster, labels, pois, means):
    labeled_pois = [[] for i in xrange(n_cluster)]
    for i, label in enumerate(labels):
        labeled_pois[label].append(pois[i])
    dis_std = []
    for i in xrange(n_cluster):
        distances = []
        for geo_loc in labeled_pois[i]:
            distances.append(GreatCircleDistance(geo_loc, means[i]).kilometers)
        dis_std.append(np.std(distances))
    return dis_std


def deterClusterRel(pois, means):
    labels = []
    for poi in pois:
        label = 0
        min_dis = 1e9
        for i, mean in enumerate(means):
            dis = np.sum((poi-mean)**2)
            if dis < min_dis:
                min_dis = dis
                label = i
        labels.append(label)
    return labels


GAUSSIAN_CONSTANT1 = 1./(2*np.pi)
def gaussianProb(mean, variance, x):
    a = 1./np.sqrt(variance[0]*variance[1])
    b = np.exp(-np.sum((x-mean)**2/variance)/2)
    return GAUSSIAN_CONSTANT1*a*b


GAUSSIAN_CONSTANT2 = 1./np.sqrt(2*np.pi)
def normGaussian(x):
    return GAUSSIAN_CONSTANT2*np.exp(-x**2/2)


def kernelInfluence(mean, dis_std, x):
    #dis = np.sqrt(np.sum((x-mean)**2))
    dis = GreatCircleDistance(x, mean).kilometers
    return 1./dis_std*normGaussian(dis/dis_std)


def personalKernelInfluence(means, geoaddr):
    dis = np.sqrt(np.sum((geoaddr-means)**2,1))
    dis_std = np.std(dis)
    factor = []
    for i in xrange(len(means)):
        factor.append(1./dis_std*normGaussian(dis[i]/dis_std))
        #factor.append(normGaussian(dis[i]/dis_var))
    return factor


def distanceKernelInfluence(means, geoaddr):
    dis = []
    for mean in means:
        dis.append(float(GreatCircleDistance(mean, geoaddr).kilometers))
    factor = []
    for i in xrange(len(means)):
        factor.append(1./DIS_STD*normGaussian(dis[i]/DIS_STD))
    return factor


def outputCenterforVis(means):
    vis_path = "./centent-vis.txt"
    means = [str(entry[0])+" "+str(entry[1]) for entry in means]
    output_str = ",".join(means)
    wfd = open(vis_path, "w")
    wfd.write("%s\n" % output_str)
    wfd.close()


def showNumInEachCluster(labels, n_cluster):
    num_in_cluster = [0 for i in xrange(n_cluster)]
    for label in labels:
        num_in_cluster[label] += 1
    for i, num in enumerate(num_in_cluster):
        print 'Cluster %d: %d' % (i+1, num)


def smoothVar(variances):
    for i in xrange(len(variances)):
        variances[i] = variances[i] + MIN_EPSILON
    return variances


def normalize(values):
    norm = np.sum(values)
    if norm == 0:
        #print values
        #raw_input("Normalization is 0")
        globals()['num_zero'] += 1
        norm_values = [0.0 for i in xrange(len(values))]
    else:
        norm_values = values/np.sum(values)
    return norm_values


def roundToZero(var):
    for i in xrange(len(var)):
        if var[i] < MIN_EPSILON:
            var[i] = 0
    return var


def projectOper(var):
    for i in xrange(len(var)):
        if var[i] < 0:
            var[i] = 0
    return var


class GeoMF_D():
    def __init__(self, cluster_method=2, cluter_tag=False, train_path=None, event_info_path=None, city_id=None):
        self.init_choice = 1      # 0:zero; 1:gaussian
        self.loss_choice = 0      # 0:reg; 1:pairwise ranking
        self.geo_influence_choice = 1
        self.ndim = 20
        self.tr_method = 0        # 0:SGD1; 1:SGD2
        self.cluster_method = cluster_method   # 0:DPGMM; 1:GMM; 2:K-means
        self.n_components = 50
        self.city_id = city_id

        # SGD
        self.niters1 = 20
        self.lr1 = 0.01
        self.lambda1 = 0.001
        self.neg_num1 = 5
        self.beta1 = 1
        self.alpha1 = 1
        self.ins_weight = [self.beta1, self.alpha1]

        pois = []
        if cluter_tag == True:
            events = set([entry[1] for entry in csv.reader(open(train_path, "r"))])
            for entry in csv.reader(open(event_info_path, "r")):
                event = entry[0]
                if event in events:
                    poi = map(float, entry[3].split(" "))
                    pois.append(poi)
                    if not checkGeoScope(poi, self.city_id):
                        print 'Invalic location'
                        sys.exit(1)
            if self.cluster_method == 0:
                cluster = DPGMM(n_components=500,
                                covariance_type='diag',
                                alpha=1,
                                n_iter=50)
                cluster.fit(pois)
                centers = removeDup(cluster.means_)
                outputCenterforVis(centers)
                self.n_components = len(centers)
                cluster_fd = open(settings["DPGMM_CLUSTER"], "wb")
                pickle.dump([centers, None], cluster_fd)
                outputCenterforVis(centers)
            elif self.cluster_method == 1:
                cluster = GMM(n_components = self.n_components,
                              covariance_type='diag',
                              min_covar=1e-7,
                              n_init=10,
                              random_state=0,
                              n_iter=100)
                cluster.fit(pois)
                outputCenterforVis(cluster.means_)
                labels = deterClusterRel(pois, cluster.means_)
                #covars = smoothVar(cluster.covars_)
                #showNumInEachCluster(labels, self.n_components)
                #dis_variances = calDisVariance(self.n_components, labels, pois)
                #dis_variances = smoothVar(dis_variances)
                dis_std = calDistanceStd(self.n_components, labels, pois, cluster.means_)
                cluster_fd = open(settings["GMM_CLUSTER"], "wb")
                pickle.dump([cluster.means_, cluster.covars_, dis_std], cluster_fd)
            elif self.cluster_method == 2:
                cluster = KMeans(n_clusters = self.n_components,
                                 max_iter=300,
                                 init='k-means++')
                cluster.fit(pois)
                means, variances= calCenterCov(self.n_components, cluster.labels_, pois)
                variances = smoothVar(variances)
                outputCenterforVis(means)
                #dis_variances = calDisVariance(self.n_components, cluster.labels_, pois)
                #dis_variances = smoothVar(dis_variances)
                dis_std = calDistanceStd(self.n_components, cluster.labels_, pois, means)
                cluster_fd = open(settings["KMEANS_CLUSTER"], "wb")
                pickle.dump([means, variances, dis_std], cluster_fd)
            else:
                print 'Invalid choice of clustering method'
                sys.exit(1)


    def model_init(self, train_file, test_file, eventinfo_file):
        self.user_ids = {}
        self.ruser_ids = {}
        self.event_ids = {}
        self.revent_ids = {}
        data = [entry for entry in csv.reader(open(train_file, "r"))]
        for entry in data:
            uname, eventname = entry[0], entry[1]
            if uname not in self.user_ids:
                self.user_ids[uname] = len(self.user_ids)
                self.ruser_ids[self.user_ids[uname]] = uname
            if eventname not in self.event_ids:
                self.event_ids[eventname] = len(self.event_ids)
                self.revent_ids[self.event_ids[eventname]] = eventname
        self.old_event_num = len(self.event_ids)
        for entry in csv.reader(open(test_file, "r")):
            eventname = entry[1]
            if eventname not in self.event_ids:
                self.event_ids[eventname] = len(self.event_ids)
                self.revent_ids[self.event_ids[eventname]] = eventname
        self.event_addrs = [[] for i in xrange(len(self.event_ids))]
        for entry in csv.reader(open(eventinfo_file, "r"), lineterminator="\n"):
            eventname = entry[0]
            if eventname in self.event_ids:
                geoaddr = map(float, entry[3].split(' '))
                self.event_addrs[self.event_ids[eventname]] = geoaddr

        if self.cluster_method == 0:
            cluster_fd = open(settings["DPGMM_CLUSTER"], "rb")
        elif self.cluster_method == 1:
            cluster_fd = open(settings["GMM_CLUSTER"], "rb")
        elif self.cluster_method == 2:
            cluster_fd = open(settings["KMEANS_CLUSTER"], "rb")
        else:
            print 'Invalid setting of cluster_fd'
            sys.exit(1)
        means, variances, dis_std = pickle.load(cluster_fd)
        self.user_geo_factor = np.array([rPosGaussian(self.n_components) for i in xrange(len(self.user_ids))])
        if self.geo_influence_choice == 1:
            self.event_geo_factor = np.array(self.calGeoInfluence(means, variances, self.event_addrs, 1))
        elif self.geo_influence_choice == 2:
            self.event_geo_factor = np.array(self.calGeoInfluence(means, dis_std, self.event_addrs, 2))
        else:
            print "Invalid choice of geo influence choice"
            sys.exit(1)
        #self.event_geo_factor = np.array(self.calGeoInfluence(means, None, self.event_geo, 3))

        if self.tr_method == 0:
            self.pool_eids = [i for i in xrange(self.old_event_num)]
            self.tr_pairs = [[self.user_ids[entry[0]], self.event_ids[entry[1]]] for entry in data]
            self.user_interacted_event = defaultdict(set)
            for entry in data:
                uid, eid = self.user_ids[entry[0]], self.event_ids[entry[1]]
                self.user_interacted_event[uid].add(eid)
        else:
            print 'Invalid settings of training method'
            sys.exit(1)
        del data

        print "Number of users: %d" % len(self.user_ids)
        #print "Number of organizers: %d" % len(self.organizer_ids)
        print "Number of events: %d" % len(self.event_ids)
        print "Number of revents: %d" % len(self.revent_ids)
        print "Number of old events: %d" % self.old_event_num
        #raw_input()


    def calGeoInfluence(self, means, variances, geoaddrs, method):
        globals()["num_zero"] = 0
        event_geo_factor = []
        if method == 1:
            for geoaddr in geoaddrs:
                geo_factor = []
                for i, mean in enumerate(means):
                    geo_factor.append(gaussianProb(mean, variances[i], geoaddr))
                #print geo_factor
                #raw_input()
                geo_factor = normalize(roundToZero(geo_factor))
                #geo_factor = roundToZero(geo_factor)
                #print geo_factor
                #raw_input("Gaussian influence above.")
                event_geo_factor.append(geo_factor)
            if globals()["num_zero"] > 0:
                print "Zero num of geo factor: %d" % globals()["num_zero"]
                raw_input()
        elif method == 2:
            for geoaddr in geoaddrs:
                geo_factor = []
                for i, mean in enumerate(means):
                    geo_factor.append(kernelInfluence(mean, variances[i], geoaddr))
                geo_factor = roundToZero(geo_factor)
                #print geo_factor
                #raw_input("Kernel influence above.")
                event_geo_factor.append(geo_factor)
        else:
            for geoaddr in geoaddrs:
                geo_factor = normalize(roundToZero(personalKernelInfluence(means, geoaddr)))
                print geo_factor
                raw_input("Personalized kernel influence factor above")
                event_geo_factor.append(geo_factor)
        return event_geo_factor


    def train(self):
        print 'Start training'
        self.evaluation(True)
        if self.loss_choice == 0:
            if self.tr_method == 0:
                for i in xrange(self.niters1):
                    tic()
                    self.sgd1_train()
                    cost = toc()
                    print 'Iteration %d, time cost: %.3f seconds.' % (i+1, cost)
                    #self.evaluation(True)
            else:
                print 'Invalid settings of training method'
                sys.exit(1)
        elif self.loss_choice == 1:
            pass
        else:
            print 'Invalid choice of loss function'
        self.evaluation(True)


    def sgd1_train(self):
        random.shuffle(self.tr_pairs)
        scan_idx = 0
        for i, pair in enumerate(self.tr_pairs):
            uid, eid = pair
            self.sgd_update(uid, eid, 1, self.alpha1)
            finished_neg_num = 0
            for j in xrange(scan_idx, len(self.pool_eids)):
                neg_eid = self.pool_eids[j]
                if neg_eid not in self.user_interacted_event[uid]:
                    self.sgd_update(uid, neg_eid, 0, self.beta1)
                    finished_neg_num += 1
                    if finished_neg_num == self.neg_num1:
                        break
            scan_idx = j+1
            if scan_idx == len(self.pool_eids):
                scan_idx = 0
                random.shuffle(self.pool_eids)
            if (i+1) % 1000 == 0:
                sys.stdout.write("\rFINISHED PAIR NUM: %d. " % (i+1))
                sys.stdout.flush()


    def sgd_update(self, uid, eid, label, weight):
        res = np.dot(self.user_geo_factor[uid], self.event_geo_factor[eid])-label
        self.user_geo_factor[uid] -= self.lr1*(weight*res*self.event_geo_factor[eid]+self.lambda1*self.user_geo_factor[uid])
        #self.user_geo_factor[uid] = projectOper(self.user_geo_factor[uid])


    def evaluation(self, weight_tag = False):
        total_error = 0.0
        if weight_tag:
            '''for uid in xrange(len(self.user_ids)):
                for eid in xrange(self.old_event_num):
                    total_error += self.ins_weight[self.user_pref[uid][eid]]*(np.dot(self.user_geo_factor[uid], self.event_geo_factor[eid])-self.user_pref[uid][eid])**2
                sys.stdout.write("\rFINISHED USER NUM: %d. " % (uid+1))
                sys.stdout.flush()'''
            for uid in xrange(len(self.user_ids)):
                for eid in xrange(self.old_event_num):
                    if eid in self.user_interacted_event[uid]:
                        total_error += self.ins_weight[1]*(np.dot(self.user_geo_factor[uid], self.event_geo_factor[eid])-1)**2
                    else:
                        total_error += self.ins_weight[0]*(np.dot(self.user_geo_factor[uid], self.event_geo_factor[eid])-0)**2
                sys.stdout.write("\rFINISHED USER NUM: %d. " % (uid+1))
                sys.stdout.flush()
        else:
            result = (np.dot(self.user_geo_factor, np.transpose(self.event_geo_factor[:self.old_event_num]))-self.user_pref[:][:self.old_event_num])**2
            total_error = np.sum(result)
        error = total_error/(len(self.user_ids)*self.old_event_num)
        print 'Error: %f' % error


    def genRecommendResult(self, test_file, result_path):
        print "Number of test event %d" % (len(self.event_ids)-self.old_event_num)
        wfd = open(result_path, 'w')
        for uid in xrange(len(self.user_ids)):
            wfd.write("%s" % self.ruser_ids[uid])
            newevent_pref = []
            for eid in xrange(self.old_event_num, len(self.event_ids)):
                newevent_pref.append([self.revent_ids[eid], np.dot(self.user_geo_factor[uid], self.event_geo_factor[eid])])
            results = sorted(newevent_pref, key=lambda x:x[1], reverse=True)
            recommendations = [x[0] for x in results]
            for event in recommendations[:settings["RE_TOPK"]]:
                wfd.write(",%s" % event)
            wfd.write("\n")
            sys.stdout.write("\rFINISHED USER NUM: %d. " % (uid+1))
            sys.stdout.flush()
        wfd.close()
        sys.stdout.write("\n")


class GeoMF_O():
    def __init__(self, city_id=None):
        self.tr_method = 0
        self.loss_choice = 0
        self.city_id = city_id

        # SGD
        self.niters1 = 20
        self.lr1 = 0.01
        self.lambda1 = 0.001
        self.neg_num1 = 5
        self.beta1 = 1
        self.alpha1 = 1
        self.ins_weight = [self.beta1, self.alpha1]

        self.genGrids()
        self.ndim = self.x_num*self.y_num
        print 'Number of grids %d' % self.ndim


    def genGrids(self):
        if self.city_id == 0:
            self.x_num = int((settings["GEO_SCOPE"][self.city_id][1][1]-settings["GEO_SCOPE"][self.city_id][0][1])/settings["GRID_SIZE"][self.city_id][1]+1)
            self.y_num = int((settings["GEO_SCOPE"][self.city_id][1][0]-settings["GEO_SCOPE"][self.city_id][0][0])/settings["GRID_SIZE"][self.city_id][0]+1)
            self.grids = np.array([[0.0, 0.0] for i in xrange(self.x_num*self.y_num)])
            for i in xrange(self.x_num):
                for j in xrange(self.y_num):
                    self.grids[i+j*self.x_num] = np.array(settings["GEO_SCOPE"][self.city_id][0])+np.array([j*settings["GRID_SIZE"][self.city_id][0], i*settings["GRID_SIZE"][self.city_id][1]])
        elif self.city_id == 1:
            pass


    def model_init(self, train_file, test_file, eventinfo_file):
        self.user_ids = {}
        self.ruser_ids = {}
        self.event_ids = {}
        self.revent_ids = {}
        data = [entry for entry in csv.reader(open(train_file))]
        for entry in data:
            uname, eventname = entry[0], entry[1]
            if uname not in self.user_ids:
                self.user_ids[uname] = len(self.user_ids)
                self.ruser_ids[self.user_ids[uname]] = uname
            if eventname not in self.event_ids:
                self.event_ids[eventname] = len(self.event_ids)
                self.revent_ids[self.event_ids[eventname]] = eventname
        self.old_event_num = len(self.event_ids)
        for entry in csv.reader(open(test_file, "r")):
            eventname = entry[1]
            if eventname not in self.event_ids:
                self.event_ids[eventname] = len(self.event_ids)
                self.revent_ids[self.event_ids[eventname]] = eventname
        self.event_addrs = [[] for i in xrange(len(self.event_ids))]
        for entry in csv.reader(open(eventinfo_file, "r"), lineterminator="\n"):
            eventname = entry[0]
            if eventname in self.event_ids:
                geoaddr = map(float, entry[3].split(' '))
                self.event_addrs[self.event_ids[eventname]] = geoaddr

        self.user_geo_factor = np.array([rPosGaussian(self.ndim) for i in
            xrange(len(self.user_ids))])
        self.event_geo_factor = np.array(self.calGeoInfluence())

        if self.tr_method == 0:
            self.pool_eids = [i for i in xrange(self.old_event_num)]
            self.tr_pairs = [[self.user_ids[entry[0]], self.event_ids[entry[1]]] for entry in data]
            self.user_interacted_event = defaultdict(set)
            for entry in data:
                uid, eid = self.user_ids[entry[0]], self.event_ids[entry[1]]
                self.user_interacted_event[uid].add(eid)
        else:
            print 'Invalid settings of training method'
            sys.exit(1)

        '''self.user_pref = {}
        for uid in xrange(len(self.user_ids)):
            self.user_pref[uid] = np.array([0 for i in xrange(len(self.event_ids))])
        for entry in data:
            uid, eid= self.user_ids[entry[0]], self.event_ids[entry[1]]
            self.user_pref[uid][eid] = 1'''

        del data
        print "Number of users: %d" % len(self.user_ids)
        #print "Number of organizers: %d" % len(self.organizer_ids)
        print "Number of events: %d" % len(self.event_ids)


    def calGeoInfluence(self):
        event_geo_factor = []
        for geo_addr in self.event_addrs:
            #geo_factor = roundToZero(personalKernelInfluence(self.grids, geo_addr))
            geo_factor = roundToZero(distanceKernelInfluence(self.grids, geo_addr))
            event_geo_factor.append(geo_factor)
        return event_geo_factor


    def train(self):
        print 'Start training'
        self.evaluation(True)
        if self.loss_choice == 0:
            if self.tr_method == 0:
                for i in xrange(self.niters1):
                    tic()
                    self.sgd1_train()
                    cost = toc()
                    print 'Iteration %d, time cost: %.3f seconds.' % (i+1, cost)
                    #self.evaluation(True)
            else:
                print 'Invalid settings of training method'
                sys.exit(1)
        elif self.loss_choice == 1:
            pass
        else:
            print 'Invalid choice of loss function'
        self.evaluation(True)


    def sgd1_train(self):
        random.shuffle(self.tr_pairs)
        scan_idx = 0
        for i, pair in enumerate(self.tr_pairs):
            uid, eid = pair
            self.sgd_update(uid, eid, 1, self.alpha1)
            finished_neg_num = 0
            for j in xrange(scan_idx, len(self.pool_eids)):
                neg_eid = self.pool_eids[j]
                if neg_eid not in self.user_interacted_event[uid]:
                    self.sgd_update(uid, neg_eid, 0, self.beta1)
                    finished_neg_num += 1
                    if finished_neg_num == self.neg_num1:
                        break
            scan_idx = j+1
            if scan_idx == len(self.pool_eids):
                scan_idx = 0
                random.shuffle(self.pool_eids)
            if (i+1) % 1000 == 0:
                sys.stdout.write("\rFINISHED PAIR NUM: %d. " % (i+1))
                sys.stdout.flush()


    def sgd_update(self, uid, eid, label, weight):
        res = np.dot(self.user_geo_factor[uid], self.event_geo_factor[eid])-label
        self.user_geo_factor[uid] -= self.lr1*(weight*res*self.event_geo_factor[eid]+self.lambda1*self.user_geo_factor[uid])
        self.user_geo_factor[uid] = projectOper(self.user_geo_factor[uid])


    def evaluation(self, weight_tag = False):
        total_error = 0.0
        if weight_tag:
            '''for uid in xrange(len(self.user_ids)):
                for eid in xrange(self.old_event_num):
                    total_error += self.ins_weight[self.user_pref[uid][eid]]*(np.dot(self.user_geo_factor[uid], self.event_geo_factor[eid])-self.user_pref[uid][eid])**2
                sys.stdout.write("\rFINISHED USER NUM: %d. " % (uid+1))
                sys.stdout.flush()'''
            for uid in xrange(len(self.user_ids)):
                for eid in xrange(self.old_event_num):
                    if eid in self.user_interacted_event[uid]:
                        total_error += self.ins_weight[1]*(np.dot(self.user_geo_factor[uid], self.event_geo_factor[eid])-1)**2
                    else:
                        total_error += self.ins_weight[0]*(np.dot(self.user_geo_factor[uid], self.event_geo_factor[eid])-0)**2
                sys.stdout.write("\rFINISHED USER NUM: %d. " % (uid+1))
                sys.stdout.flush()
        else:
            result = (np.dot(self.user_geo_factor, np.transpose(self.event_geo_factor[:self.old_event_num]))-self.user_pref[:][:self.old_event_num])**2
            total_error = np.sum(result)
        error = total_error/(len(self.user_ids)*self.old_event_num)
        print 'Error: %f' % error


    def genRecommendResult(self, test_file, result_path):
        print "Number of test event %d" % (len(self.event_ids)-self.old_event_num)
        wfd = open(result_path, 'w')
        for uid in xrange(len(self.user_ids)):
            wfd.write("%s" % self.ruser_ids[uid])
            newevent_pref = []
            for eid in xrange(self.old_event_num, len(self.event_ids)):
                newevent_pref.append([self.revent_ids[eid], np.dot(self.user_geo_factor[uid], self.event_geo_factor[eid])])
            results = sorted(newevent_pref, key=lambda x:x[1], reverse=True)
            recommendations = [x[0] for x in results]
            for event in recommendations[:settings["RE_TOPK"]]:
                wfd.write(",%s" % event)
            wfd.write("\n")
            sys.stdout.write("\rFINISHED USER NUM: %d. " % (uid+1))
            sys.stdout.flush()
        wfd.close()
        sys.stdout.write("\n")


class HeSig():
    def __init__(self, cluster_method=2, cluter_tag=False, train_path=None, event_info_path=None, city_id=None):
        self.loss_choice = 0      # 0:reg; 1:pairwise ranking
        self.ndim = 20
        self.tr_method = 0        # 0:SGD1; 1:SGD2
        self.cluster_method = cluster_method   # 0:DPGMM; 1:GMM; 2:K-means
        self.n_components = 20
        self.city_id = city_id

        # SGD
        self.niters1 = 10
        self.lr1 = 0.01
        self.lambda1 = 0.001
        self.neg_num1 = 5
        self.beta1 = 1
        self.alpha1 = 1
        self.ins_weight = [self.beta1, self.alpha1]

        pois = []
        if cluter_tag == True:
            events = set([entry[1] for entry in csv.reader(open(train_path, "r"))])
            for entry in csv.reader(open(event_info_path, "r")):
                event = entry[0]
                if event in events:
                    poi = map(float, entry[3].split(" "))
                    pois.append(poi)
                    if not checkGeoScope(poi, self.city_id):
                        print 'Invalic location'
                        sys.exit(1)
            if self.cluster_method == 0:
                cluster = DPGMM(n_components=500,
                                covariance_type='diag',
                                alpha=1,
                                n_iter=50)
                cluster.fit(pois)
                centers = removeDup(cluster.means_)
                outputCenterforVis(centers)
                self.n_components = len(centers)
                cluster_fd = open(settings["DPGMM_CLUSTER"], "wb")
                pickle.dump([centers, None], cluster_fd)
                self.model_path = settings["GEOMF"]
                outputCenterforVis(centers)
            elif self.cluster_method == 1:
                cluster = GMM(n_components = self.n_components,
                              covariance_type='diag',
                              min_covar=1e-7,
                              n_init=10,
                              random_state=0,
                              n_iter=100)
                cluster.fit(pois)
                outputCenterforVis(cluster.means_)
                labels = deterClusterRel(pois, cluster.means_)
                #showNumInEachCluster(labels, self.n_components)
                dis_variances = calDisVariance(self.n_components, labels, pois)
                dis_variances = smoothVar(dis_variances)
                covars = smoothVar(cluster.covars_)
                cluster_fd = open(settings["GMM_CLUSTER"], "wb")
                pickle.dump([cluster.means_, covars, dis_variances], cluster_fd)
            elif self.cluster_method == 2:
                cluster = KMeans(n_clusters = self.n_components,
                                 max_iter=300,
                                 init='k-means++')
                cluster.fit(pois)
                means, variances= calCenterCov(self.n_components, cluster.labels_, pois)
                outputCenterforVis(means)
                dis_variances = calDisVariance(self.n_components, cluster.labels_, pois)
                variances = smoothVar(variances)
                dis_variances = smoothVar(dis_variances)
                cluster_fd = open(settings["KMEANS_CLUSTER"], "wb")
                pickle.dump([means, variances, dis_variances], cluster_fd)
            else:
                print 'Invalid choice of clustering method'
                sys.exit(1)


    def model_init(self, train_file, test_file, eventinfo_file):
        self.user_ids = {}
        self.ruser_ids = {}
        self.event_ids = {}
        self.revent_ids = {}
        data = [entry for entry in csv.reader(open(train_file))]
        for entry in data:
            uname, eventname = entry[0], entry[1]
            if uname not in self.user_ids:
                self.user_ids[uname] = len(self.user_ids)
                self.ruser_ids[self.user_ids[uname]] = uname
            if eventname not in self.event_ids:
                self.event_ids[eventname] = len(self.event_ids)
                self.revent_ids[self.event_ids[eventname]] = eventname
        self.old_event_num = len(self.event_ids)
        for entry in csv.reader(open(test_file, "r")):
            eventname = entry[1]
            if eventname not in self.event_ids:
                self.event_ids[eventname] = len(self.event_ids)
                self.revent_ids[self.event_ids[eventname]] = eventname
        self.event_addrs = [[] for i in xrange(len(self.event_ids))]
        for entry in csv.reader(open(eventinfo_file, "r"), lineterminator="\n"):
            eventname = entry[0]
            if eventname in self.event_ids:
                geoaddr = map(float, entry[3].split(' '))
                self.event_addrs[self.event_ids[eventname]] = geoaddr

        if self.cluster_method == 0:
            cluster_fd = open(settings["DPGMM_CLUSTER"], "rb")
        elif self.cluster_method == 1:
            cluster_fd = open(settings["GMM_CLUSTER"], "rb")
        elif self.cluster_method == 2:
            cluster_fd = open(settings["KMEANS_CLUSTER"], "rb")
        else:
            print 'Invalid setting of cluster_fd'
            sys.exit(1)
        means, variances, dis_variances = pickle.load(cluster_fd)
        self.user_geo_factor = np.array([rPosGaussian(self.ndim) for i in
            xrange(len(self.user_ids))])
        self.grid_geo_factor = np.array([rPosGaussian(self.ndim) for i in
            xrange(self.n_components)])
        self.event_grid_dis = np.array(self.calEventGridDis(means, variances, self.event_addrs))

        if self.tr_method == 0:
            self.pool_eids = [i for i in xrange(self.old_event_num)]
            self.tr_pairs = [[self.user_ids[entry[0]], self.event_ids[entry[1]]] for entry in data]
            self.user_interacted_event = defaultdict(set)
            for entry in data:
                uid, eid = self.user_ids[entry[0]], self.event_ids[entry[1]]
                self.user_interacted_event[uid].add(eid)
        else:
            print 'Invalid settings of training method'
            sys.exit(1)

        '''self.user_pref = {}
        for uid in xrange(len(self.user_ids)):
            self.user_pref[uid] = np.array([0 for i in xrange(len(self.event_ids))])
        for entry in data:
            uid, eid= self.user_ids[entry[0]], self.event_ids[entry[1]]
            self.user_pref[uid][eid] = 1'''

        del data
        print "Number of users: %d" % len(self.user_ids)
        #print "Number of organizers: %d" % len(self.organizer_ids)
        print "Number of events: %d" % len(self.event_ids)


    def calEventGridDis(self, means, variances, geoaddrs):
        globals()["num_zero"] = 0
        event_grid_dis = []
        for geoaddr in geoaddrs:
            grid_dis = []
            for i, mean in enumerate(means):
                grid_dis.append(gaussianProb(mean, variances[i], geoaddr))
            grid_dis = normalize(roundToZero(grid_dis))
            event_grid_dis.append(grid_dis)
        if globals()["num_zero"] > 0:
            print "Zero num of geo factor: %d" % globals()["num_zero"]
            raw_input()
        return event_grid_dis


    def train(self):
        print 'Start training'
        #self.evaluation()
        if self.loss_choice == 0:
            if self.tr_method == 0:
                for i in xrange(self.niters1):
                    tic()
                    self.sgd1_train()
                    cost = toc()
                    print 'Iteration %d, time cost: %.3f seconds.' % (i+1, cost)
                    #self.evaluation(True)
            else:
                print 'Invalid settings of training method'
                sys.exit(1)
        elif self.loss_choice == 1:
            pass
        else:
            print 'Invalid choice of loss function'
        #self.evaluation()


    def sgd1_train(self):
        random.shuffle(self.tr_pairs)
        scan_idx = 0
        for i, pair in enumerate(self.tr_pairs):
            uid, eid = pair
            self.sgd_update(uid, eid, 1, self.alpha1)
            finished_neg_num = 0
            for j in xrange(scan_idx, len(self.pool_eids)):
                neg_eid = self.pool_eids[j]
                if neg_eid not in self.user_interacted_event[uid]:
                    self.sgd_update(uid, neg_eid, 0, self.beta1)
                    finished_neg_num += 1
                    if finished_neg_num == self.neg_num1:
                        break
            scan_idx = j+1
            if scan_idx == len(self.pool_eids):
                scan_idx = 0
                random.shuffle(self.pool_eids)
            if (i+1) % 1000 == 0:
                sys.stdout.write("\rFINISHED PAIR NUM: %d. " % (i+1))
                sys.stdout.flush()


    def sgd_update(self, uid, eid, label, weight):
        pred = 0.0
        for i in xrange(self.n_components):
            pred += np.dot(self.user_geo_factor[uid], self.grid_geo_factor[i])*self.event_grid_dis[eid][i]
        res = pred - label
        grad_user_factor = np.zeros(self.ndim)
        for i in xrange(self.n_components):
            grad_user_factor += weight*res*self.grid_geo_factor[i]*self.event_grid_dis[eid][i]
            self.grid_geo_factor[i] -= self.lr1*(weight*res*self.user_geo_factor[uid]*self.event_grid_dis[eid][i]+self.lambda1*self.grid_geo_factor[i])
        self.user_geo_factor[uid] -= self.lr1*(grad_user_factor+self.lambda1*self.user_geo_factor[uid])


    def evaluation(self):
        total_error = 0.0
        for uid in xrange(len(self.user_ids)):
            for eid in xrange(self.old_event_num):
                pred = 0.0
                for i in xrange(self.n_components):
                    pred += np.dot(self.user_geo_factor[uid], self.grid_geo_factor[i])*self.event_grid_dis[eid][i]
                if eid in self.user_interacted_event[uid]:
                    total_error += self.ins_weight[1]*(pred-1)**2
                else:
                    total_error += self.ins_weight[0]*(pred-0)**2
            sys.stdout.write("\rFINISHED USER NUM: %d. " % (uid+1))
            sys.stdout.flush()
        error = total_error/(len(self.user_ids)*self.old_event_num)
        print 'Error: %f' % error


    def genRecommendResult(self, test_file, result_path):
        print "Number of test event %d" % (len(self.event_ids)-self.old_event_num)
        wfd = open(result_path, 'w')
        for uid in xrange(len(self.user_ids)):
            wfd.write("%s" % self.ruser_ids[uid])
            newevent_pref = []
            for eid in xrange(self.old_event_num, len(self.event_ids)):
                pred = 0.0
                for i in xrange(self.n_components):
                    pred += np.dot(self.user_geo_factor[uid], self.grid_geo_factor[i])*self.event_grid_dis[eid][i]
                newevent_pref.append([self.revent_ids[eid], pred])
            results = sorted(newevent_pref, key=lambda x:x[1], reverse=True)
            recommendations = [x[0] for x in results]
            for event in recommendations[:settings["RE_TOPK"]]:
                wfd.write(",%s" % event)
            wfd.write("\n")
            sys.stdout.write("\rFINISHED USER NUM: %d. " % (uid+1))
            sys.stdout.flush()
        wfd.close()
        sys.stdout.write("\n")


class Distance():
    def __init__(self, city_id):
        self.city_id = city_id
        self.alpha1 = 1
        self.beta1 = 1
        # SGD1
        self.niters1 = 5
        self.lr1 = 0.001
        self.lambda1 = 0.001
        self.w0 = rPosGaussian(1)[0]
        self.w1 = rPosGaussian(1)[0]
        self.a = 0.0
        self.b = 0.0

    def model_init(self, train_file, test_file, eventinfo_file):
        self.user_ids = {}
        self.ruser_ids = {}
        self.event_ids = {}
        self.revent_ids = {}
        data = [entry for entry in csv.reader(open(train_file))]
        for entry in data:
            uname, eventname = entry[0], entry[1]
            if uname not in self.user_ids:
                self.user_ids[uname] = len(self.user_ids)
                self.ruser_ids[self.user_ids[uname]] = uname
            if eventname not in self.event_ids:
                self.event_ids[eventname] = len(self.event_ids)
                self.revent_ids[self.event_ids[eventname]] = eventname
        self.old_event_num = len(self.event_ids)
        for entry in csv.reader(open(test_file, "r")):
            eventname = entry[1]
            if eventname not in self.event_ids:
                self.event_ids[eventname] = len(self.event_ids)
                self.revent_ids[self.event_ids[eventname]] = eventname
        self.event_addrs = [[] for i in xrange(len(self.event_ids))]
        for entry in csv.reader(open(eventinfo_file, "r"), lineterminator="\n"):
            eventname = entry[0]
            if eventname in self.event_ids:
                geoaddr = map(float, entry[3].split(' '))
                self.event_addrs[self.event_ids[eventname]] = geoaddr

        self.pool_eids = [i for i in xrange(self.old_event_num)]
        self.user_interacted_event = defaultdict(set)
        for entry in data:
            uid, eid = self.user_ids[entry[0]], self.event_ids[entry[1]]
            self.user_interacted_event[uid].add(eid)
        self.tr_instances = []
        for uid in self.user_interacted_event:
            events = list(self.user_interacted_event[uid])
            random.shuffle(events)
            for i in xrange(len(events[:5])):
                for j in xrange(i+1, len(events[:5])):
                    distance = GreatCircleDistance(self.event_addrs[events[i]], self.event_addrs[events[j]]).kilometers
                    if distance < MIN_EPSILON:
                        self.tr_instances.append([uid, events[i], events[j], -23])
                    else:
                        self.tr_instances.append([uid, events[i], events[j], np.log(distance)])
            sys.stdout.write("\rFINISHED UID NUM: %d. " % (uid+1))
            sys.stdout.flush()


        del data
        print "Number of users: %d" % len(self.user_ids)
        #print "Number of organizers: %d" % len(self.organizer_ids)
        print "Number of events: %d" % len(self.event_ids)


    def train(self):
        print 'Start training'
        for i in xrange(self.niters1):
            tic()
            self.sgd1_train()
            cost = toc()
            print 'Iteration %d, time cost: %.3f seconds.' % (i+1, cost)


    def sgd1_train(self):
        random.shuffle(self.tr_instances)
        scan_idx = 0
        for i, instance in enumerate(self.tr_instances):
            uid, eid1, eid2, distance = instance
            self.sgd_update(distance, 0, self.alpha1)
            for j in xrange(scan_idx, len(self.pool_eids)):
                neg_eid = self.pool_eids[j]
                if neg_eid not in self.user_interacted_event[uid]:
                    distance = GreatCircleDistance(self.event_addrs[eid1], self.event_addrs[neg_eid]).kilometers
                    if distance < MIN_EPSILON:
                        distance = -23
                    else:
                        distance = np.log(distance)
                    self.sgd_update(distance, -23, self.beta1)
                    distance = GreatCircleDistance(self.event_addrs[eid2], self.event_addrs[neg_eid]).kilometers
                    if distance < MIN_EPSILON:
                        distance = -23
                    else:
                        distance = np.log(distance)
                    self.sgd_update(distance, -23, self.beta1)
                    break
            scan_idx = j+1
            if scan_idx == len(self.pool_eids):
                scan_idx = 0
                random.shuffle(self.pool_eids)
            if (i+1) % 1000 == 0:
                sys.stdout.write("\rFINISHED PAIR NUM: %d. " % (i+1))
                sys.stdout.flush()
        self.a = 2**self.w0
        self.b = self.w1


    def sgd_update(self, distance, label, weight):
        y = self.predict(distance, True)
        self.w0 -= self.lr1*(weight*(y-label)+self.lambda1*self.w0)
        self.w1 -= self.lr1*(weight*(y-label)*distance+self.lambda1*self.w1)


    def predict(self, distance, log_tag=False):
        if log_tag:
            y = self.w0+self.w1*distance
        else:
            y = self.a*distance**self.b
        return y


    def genRecommendResult(self, test_file, result_path):
        print "Number of test event %d" % (len(self.event_ids)-self.old_event_num)
        wfd = open(result_path, 'w')
        for uid in xrange(len(self.user_ids)):
            wfd.write("%s" % self.ruser_ids[uid])
            newevent_pref = []
            for eid in xrange(self.old_event_num, len(self.event_ids)):
                pred = 0.0
                for attended_eid in self.user_interacted_event[uid]:
                    pred += self.predict(GreatCircleDistance(self.event_addrs[eid], self.event_addrs[attended_eid]).kilometers, True)
                newevent_pref.append([self.revent_ids[eid], pred])
            results = sorted(newevent_pref, key=lambda x:x[1], reverse=True)
            recommendations = [x[0] for x in results]
            for event in recommendations[:settings["RE_TOPK"]]:
                wfd.write(",%s" % event)
            wfd.write("\n")
            sys.stdout.write("\rFINISHED USER NUM: %d. " % (uid+1))
            sys.stdout.flush()
        wfd.close()
        sys.stdout.write("\n")

