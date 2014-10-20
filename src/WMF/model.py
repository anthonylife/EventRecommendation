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
###################################################################

import numpy as np
import json, csv, sys, random, pickle
from collections import defaultdict
from tool import rZero, rPosGaussian

settings = json.loads(open("../../SETTINGS.json").read())

MIN_PREF = -1e5

class WMF():
    def __init__(self):
        self.niters = 20
        self.ndim = 20
        self.lda = 1
        self.alpha = 40

    def model_init(self, train_file, init_choice):
        self.user_ids = {}
        self.ruser_ids = {}
        self.organizer_ids = {}
        data = [entry for entry in csv.reader(open(train_file))]

        for entry in data:
            uname, oname = entry[0], entry[3]
            if uname not in self.user_ids:
                self.user_ids[uname] = len(self.user_ids)
                self.ruser_ids[self.user_ids[uname]] = uname
            if oname not in self.organizer_ids:
                self.organizer_ids[oname] = len(self.organizer_ids)

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

        self.user_pref = {}
        self.user_conf = {}
        for uid in self.user_ids:
            self.user_pref[uid] = np.array([0 for i in xrange(len(self.organizer_ids))])
            self.user_conf[uid] = np.array([1 for i in xrange(len(self.organizer_ids))])
        for entry in data:
            uid, oid = self.user_ids[entry[0]], self.organizer_ids[entry[3]]
            self.user_pref[uid][oid] = 1
            self.user_conf[uid][oid] += self.alpha
        del data

        self.organizer_pref = {}
        self.organizer_conf = {}
        for oid in self.organizer_ids:
            self.organizer_pref[oid] = np.array([0 for i in xrange(len(self.user_ids))])
            self.organizer_conf[oid] = np.array([0 for i in xrange(len(self.user_ids))])
        for uid in self.user_ids:
            for oid in self.organizer_ids:
                self.organizer_pref[oid][uid] = self.user_pref[uid][oid]
                self.organizer_conf[oid][uid] = self.user_conf[uid][oid]
        print "Number of users: %d" % len(user_ids)
        print "Number of organizers: %d" % len(organizer_ids)


    def train(self, lr_method):
        print 'Start training'
        for i in xrange(self.niters):
            self.als_train()
            print 'Iteration %d' % (i+1)
        self.save_model()


    def als_train(self):
        for uid in xrange(len(self.user_ids)):
            sys.stdout.write("\rFINISHED UID NUM: %d. " % (uid+1))
            sys.stdout.flush()
            self.user_factor[uid] = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(np.transpose(self.organizer_factor), np.diag(self.user_conf[uid])), self.organizer_factor)+self.lda*np.eye(self.ndim)), np.transpose(self.organizer_factor)), np.diag(self.user_conf[uid])), np.transpose(self.user_pref[uid]))
        for oid in xrange(len(self.organizer_ids)):
            sys.stdout.write("\rFINISHED OID NUM: %d. " % (oid+1))
            sys.stdout.flush()
            self.organizer_factor[oid] = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(np.transpose(self.user_factor),np.diag(self.organizer_conf[oid])), self.user_factor)+self.lda*np.eye(self.ndim)), np.transpose(self.user_factor)),np.diag(self.organizer_conf[oid])),np.transpose(self.organizer_pref[oid]))


    def evaluation(self):
        pass

    def save_model(self):
        model_fd = open(settings["WMF_MODEL"], "wb")
        pickle.dump([self.user_factor, self.organizer_factor], model_fd)

    def load_model(self):
        model_fd = open(settings["WMF_MODEL"], "rb")
        self.user_factor, self.organizer_factor = pickle.load(model_fd)


    def genRecommendResult(self, restart, test_file, init_choice, result_path):
        if not restart:
            self.model_init(lr_method, train_file, init_choice)
            self.load_model()
        data = [entry for entry in csv.reader(open(test_file))]
        event_oid = []
        for entry in data:
            eventname, oname = entry[1], entry[3]
            if oname in self.organizer_ids:
                event_oid.append([eventname, self.organizer_ids[oname]])
            else:
                event_oid.append([eventname, -1])
        del data

        wfd = open(result_path, 'w')
        for uid in xrange(len(self.user_ids)):
            wfd.write("%s" % self.ruser_ids[uid])
            newevent_pref = []
            for entry in event_oid:
                if entry[1] == -1:
                    newevent_pref.append([entry[0], MIN_PREF])
                else:
                    newevent_pref.append([entry[0], np.dot(self.user_factor[uid], self.organizer_factor[entry[1]])])
            results = sorted(newevent_pref, key=lambda x:x[1], reverse=True)
            recommendations = [x[0] for x in results]
            for event in recommendations:
                wfd.write(",%s" % event)
            wfd.write("\n")
        wfd.close()
