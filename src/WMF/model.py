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
# Date: 2014/10/20                                                #
# Weighted Matrix Factorization for implicit feedback data        #
# ALS===>>>>                                                      #
# Result: lambda1=1; beta=1; alpha=40 =====> 0.076                #
#         lambda1=0.01; beta=0.01; alpha=1 =====> 0.048           #
#         lambda1=0.01; beta=0.1; gamma=0.5; alpha=1 ===> 0.113   #
#         lambda1=0.01; beta=1; gamma=1; alpha=1 ===> 0.079       #
# SGD===>>>>                                                      #
# Result: niters2, lambda2=0.01, neg_sample=1, beta2=1, alpha2=1  #
#           ===> 0.046                                            #
###################################################################

import numpy as np
import json, csv, sys, random, pickle
from collections import defaultdict
from tool import rZero, rPosGaussian, tic, toc

settings = json.loads(open("../../SETTINGS.json").read())

MIN_PREF = -1e5

class WMF():
    def __init__(self):
        self.ndim = 20
        self.tr_method = 4

        # ALS
        self.niters1 = 20
        self.lambda1 = 0.01
        self.beta1 = 0.1
        self.gamma1 = 0.5
        self.alpha1 = 1

        # SGD
        self.niters2 = 30
        self.lr2 = 0.01
        self.lambda2 = 0.001
        self.neg_sample = 5
        self.beta2 = 0.1
        self.alpha2 = 1

    def model_init(self, train_file, init_choice):
        self.user_ids = {}
        self.ruser_ids = {}
        self.organizer_ids = {}
        self.event_ids = {}
        data = [entry for entry in csv.reader(open(train_file))]

        for entry in data:
            uname, eventname, oname = entry[0], entry[1], entry[3]
            if uname not in self.user_ids:
                self.user_ids[uname] = len(self.user_ids)
                self.ruser_ids[self.user_ids[uname]] = uname
            if oname not in self.organizer_ids:
                self.organizer_ids[oname] = len(self.organizer_ids)
            if eventname not in self.event_ids:
                self.event_ids[eventname] = len(self.event_ids)

        factor_init_method = None
        if init_choice == "zero":
            factor_init_method = rZero
        elif init_choice == "gaussian":
            factor_init_method = rPosGaussian
        else:
            print 'Choice of model initialization error.'
            sys.exit(1)

        self.user_factor = np.array([factor_init_method(self.ndim) for i in
            xrange(len(self.user_ids))])
        self.organizer_factor = np.array([factor_init_method(self.ndim) for i in
            xrange(len(self.organizer_ids))])

        # ALS learning needed
        if self.tr_method == 1:
            self.user_pref = {}
            self.user_conf = {}
            for uid in xrange(len(self.user_ids)):
                self.user_pref[uid] = np.array([0 for i in xrange(len(self.organizer_ids))])
                self.user_conf[uid] = np.array([self.beta1 for i in xrange(len(self.organizer_ids))])
            for entry in data:
                uid, oid = self.user_ids[entry[0]], self.organizer_ids[entry[3]]
                self.user_pref[uid][oid] = 1
                if self.user_conf[uid][oid] == self.beta1:
                    self.user_conf[uid][oid] = self.gamma1
                else:
                    self.user_conf[uid][oid] += self.alpha1
            self.organizer_pref = {}
            self.organizer_conf = {}
            for oid in xrange(len(self.organizer_ids)):
                self.organizer_pref[oid] = np.array([0 for i in xrange(len(self.user_ids))])
                self.organizer_conf[oid] = np.array([0 for i in xrange(len(self.user_ids))])
            for uid in xrange(len(self.user_ids)):
                for oid in xrange(len(self.organizer_ids)):
                    self.organizer_pref[oid][uid] = self.user_pref[uid][oid]
                    self.organizer_conf[oid][uid] = self.user_conf[uid][oid]
        # SGD learning needed
        if self.tr_method == 2:
            self.pool_oids = [i for i in xrange(len(self.organizer_ids))]
            self.user_interacted_organizer = defaultdict(set)
            self.tr_pairs = [[self.user_ids[entry[0]], self.organizer_ids[entry[3]]] for entry in data]
            for i, entry in enumerate(data):
                uid, oid = self.user_ids[entry[0]], self.organizer_ids[entry[3]]
                self.user_interacted_organizer[uid].add(oid)
        # SGD1 learning needed
        if self.tr_method == 3:
            self.user_pref = {}
            self.user_conf = {}
            for uid in xrange(len(self.user_ids)):
                self.user_pref[uid] = np.array([0 for i in xrange(len(self.organizer_ids))])
                self.user_conf[uid] = np.array([self.beta1 for i in xrange(len(self.organizer_ids))])
            for entry in data:
                uid, oid = self.user_ids[entry[0]], self.organizer_ids[entry[3]]
                self.user_pref[uid][oid] = 1
                if self.user_conf[uid][oid] == self.beta1:
                    self.user_conf[uid][oid] = self.gamma1
                else:
                    self.user_conf[uid][oid] += self.alpha1
            self.tr_pairs = []
            for uid in xrange(len(self.user_ids)):
                for oid in xrange(len(self.organizer_ids)):
                        self.tr_pairs.append([uid, oid, self.user_pref[uid][oid], self.user_conf[uid][oid]])
            print 'Number of training pairs: %d' % len(self.tr_pairs)
        if self.tr_method == 4:
            self.user_pref = {}
            self.user_conf = {}
            for uid in xrange(len(self.user_ids)):
                self.user_pref[uid] = np.array([0 for i in xrange(len(self.organizer_ids))])
                self.user_conf[uid] = np.array([self.beta1 for i in xrange(len(self.organizer_ids))])
            self.user_interacted_event = defaultdict(set)
            self.event_organizer = [0 for i in xrange(len(self.event_ids))]
            self.tr_pairs = []
            for entry in data:
                uid, eid, oid = self.user_ids[entry[0]], self.event_ids[entry[1]], self.organizer_ids[entry[3]]
                self.event_organizer[eid] = oid
                self.tr_pairs.append([uid, eid])
                self.user_interacted_event[uid].add(eid)
            self.pool_eids = [i for i in xrange(len(self.event_ids))]

        del data
        print "Number of users: %d" % len(self.user_ids)
        print "Number of organizers: %d" % len(self.organizer_ids)


    def train(self):
        print 'Start training'
        if self.tr_method == 1:
            for i in xrange(self.niters1):
                tic()
                self.als_train()
                cost = toc()
                print 'Iteration %d, time cost: %.3f seconds.' % (i+1, cost)
                self.evaluation()
        elif self.tr_method == 2:
            for i in xrange(self.niters2):
                tic()
                self.sgd_train()
                cost = toc()
                print 'Iteration %d, time cost: %.3f seconds.' % (i+1, cost)
        elif self.tr_method == 3:
            self.evaluation()
            for i in xrange(self.niters2):
                tic()
                self.sgd1_train()
                cost = toc()
                print 'Iteration %d, time cost: %.3f seconds.' % (i+1, cost)
                self.evaluation()
        elif self.tr_method == 4:
            self.evaluation()
            for i in xrange(self.niters2):
                tic()
                self.sgd2_train()
                cost = toc()
                print 'Iteration %d, time cost: %.3f seconds.' % (i+1, cost)
                self.evaluation()
        else:
            print 'Error choice of training method!'
            sys.exit(1)
        self.save_model()


    def als_train(self):
        for oid in xrange(len(self.organizer_ids)):
            sys.stdout.write("\rFINISHED OID NUM: %d. " % (oid+1))
            sys.stdout.flush()
            self.organizer_factor[oid] = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(self.user_factor)*self.organizer_conf[oid], self.user_factor)+self.lambda1*np.eye(self.ndim)), np.transpose(self.user_factor))*self.organizer_conf[oid],np.transpose(self.organizer_pref[oid]))
        for uid in xrange(len(self.user_ids)):
            sys.stdout.write("\rFINISHED UID NUM: %d. " % (uid+1))
            sys.stdout.flush()
            self.user_factor[uid] = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(self.organizer_factor)*self.user_conf[uid], self.organizer_factor)+self.lambda1*np.eye(self.ndim)), np.transpose(self.organizer_factor))*self.user_conf[uid], np.transpose(self.user_pref[uid]))


    def sgd_train(self):
        random.shuffle(self.tr_pairs)
        for i, pair in enumerate(self.tr_pairs):
            uid, oid = pair
            self.sgd_update(uid, oid, 1, self.alpha2)
            finished_num = 0
            for neg_oid in self.pool_oids:
                if neg_oid not in self.user_interacted_organizer[uid]:
                    self.sgd_update(uid, neg_oid, 0, self.beta2)
                    finished_num += 1
                    if finished_num == self.neg_sample:
                        break
            if (i+1) % 100 == 0:
                random.shuffle(self.pool_oids)
                sys.stdout.write("\rFINISHED PAIR NUM: %d. " % (i+1))
                sys.stdout.flush()


    def sgd1_train(self):
        random.shuffle(self.tr_pairs)
        for i, pair in enumerate(self.tr_pairs):
            self.sgd_update(pair[0], pair[1], pair[2], 2*pair[3])
            if (i+1) % 1000 == 0:
                sys.stdout.write("\rFINISHED PAIR NUM: %d. " % (i+1))
                sys.stdout.flush()


    def sgd2_train(self):
        random.shuffle(self.tr_pairs)
        scan_idx = 0
        for i, pair in enumerate(self.tr_pairs):
            uid, eid = pair
            oid = self.event_organizer[eid]
            self.sgd_update(uid, oid, 1, self.alpha2)
            finished_neg_num = 0
            for j in xrange(scan_idx, len(self.pool_eids)):
                neg_eid = self.pool_eids[j]
                if neg_eid not in self.user_interacted_event[uid]:
                    self.sgd_update(uid, self.event_organizer[neg_eid], 0, self.beta2)
                    finished_neg_num += 1
                    if finished_neg_num == self.neg_sample:
                        break
            scan_idx = j+1
            if scan_idx == len(self.pool_eids):
                scan_idx = 0
                random.shuffle(self.pool_eids)
            if (i+1) % 1000 == 0:
                sys.stdout.write("\rFINISHED PAIR NUM: %d. " % (i+1))
                sys.stdout.flush()


    def sgd_update(self, uid, oid, label, weight):
        res = weight*(np.dot(self.user_factor[uid], self.organizer_factor[oid]) - label)
        tmp_user_factor = self.user_factor[uid] - self.lr2*(res*self.organizer_factor[oid]+self.lambda2*self.user_factor[uid])
        try:
            self.organizer_factor[oid] -= self.lr2*(res*self.user_factor[uid]+self.lambda2*self.organizer_factor[oid])
        except:
            print self.user_factor[uid]
            print self.organizer_factor[oid]
            raw_input()
        self.user_factor[uid] = tmp_user_factor


    def evaluation(self):
        total_error = 0
        for uid in xrange(len(self.user_ids)):
            for oid in xrange(len(self.organizer_ids)):
                total_error += self.user_conf[uid][oid]*(np.dot(self.user_factor[uid], self.organizer_factor[oid]) - self.user_pref[uid][oid])**2
        error = total_error/(len(self.user_ids)*len(self.organizer_ids))
        print 'Error: %f' % error


    def save_model(self):
        model_fd = open(settings["WMF_MODEL"], "wb")
        pickle.dump([self.user_factor, self.organizer_factor], model_fd)

    def load_model(self):
        model_fd = open(settings["WMF_MODEL"], "rb")
        self.user_factor, self.organizer_factor = pickle.load(model_fd)


    def genRecommendResult(self, restart, train_file, test_file, init_choice, result_path):
        if not restart:
            self.model_init(train_file, init_choice)
            self.load_model()
        data = [entry for entry in csv.reader(open(test_file))]
        event_oid = {}
        for entry in data:
            eventname, oname = entry[1], entry[3]
            if eventname in event_oid:
                continue
            if oname in self.organizer_ids:
                event_oid[eventname] = self.organizer_ids[oname]
            else:
                event_oid[eventname] = -1
        del data

        wfd = open(result_path, 'w')
        score = 0
        print 'Number of test events: %d' % len(event_oid)
        for uid in xrange(len(self.user_ids)):
            wfd.write("%s" % self.ruser_ids[uid])
            organizer_pref = {}
            newevent_pref = []
            for eventname in event_oid:
                oid = event_oid[eventname]
                if oid == -1:
                    newevent_pref.append([eventname, MIN_PREF])
                else:
                    if oid in organizer_pref:
                        newevent_pref.append([eventname, organizer_pref[oid]])
                    else:
                        score = np.dot(self.user_factor[uid], self.organizer_factor[oid])
                        organizer_pref[eventname] = score
                        newevent_pref.append([eventname, score])
            results = sorted(newevent_pref, key=lambda x:x[1], reverse=True)
            recommendations = [x[0] for x in results]
            for event in recommendations[:settings["RE_TOPK"]]:
                wfd.write(",%s" % event)
            wfd.write("\n")
            sys.stdout.write("\rFINISHED USER NUM: %d. " % (uid+1))
            sys.stdout.flush()
        wfd.close()
        print ''
