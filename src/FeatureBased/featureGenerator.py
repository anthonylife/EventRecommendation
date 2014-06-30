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
# Date: 2014/6/29                                                 #
# Feature generator: for each input pair(uid, eventid),           #
# calculating all its related features.                           #
#                                                                 #
# Types of features:                                              #
#   see corresponding feature generation functions.               #
###################################################################


import sys, csv, json, argparse, random, math
sys.path.append("../")
import numpy as np
from collections import defaultdict
from utils import getIdOfTimePeriod, isChinese, getStopwords

settings = json.loads(open("../../SETTINGS.json").read())

class FeatureGenerator():
    def __init__(self, gen_feature_method, friend_path, event_path, tr_data_path):
        self.gen_feature_method = gen_feature_method
        # for testing
        if self.gen_feature_method == 0:
            self.stopwords = self.getStopwords()
            self.user_friends = self.loadFriendship(friend_path)
            self.event_info = self.loadEventInfo(event_path)
            self.user_attend_event = self.loadUserAttendEvent(tr_data_path)
            self.organizor_hold_event = self.calOrganizorHoldEvent()
            self.user_attend_content = self.calUserContent()
            self.user_attend_category = self.calUserCategoryPref()
            self.user_attend_location = self.calUserLocation()
        # for training
        elif self.gen_feature_method == 1:
            self.stopwords = self.getStopwords()
            self.user_friendship = self.loadFriendship()
            self.event_info = self.loadEventInfo()
            self.user_attend_event = self.loadUserAttendEvent()
            self.organizor_hold_event = self.calOrganizorHoldEvent()
        else:
            print 'Invalid choice of method for generating features!'
            print '\t0: for training; 1 for testing!'
            sys.exit(1)

    def genFeature(self, uid, eventid):
        feature = []
        feature += self.genBigramFeature(uid, eventid)
        feature += self.genUnigramFeature(uid, eventid)
        return feature

    def genUnigramFeature(self, uid, eventid):
        ''' Unigram feature list:
            1.event type;
            2.length of event intro;
            3.number of entities;
            4.length of title;

        '''

    def genBigramFeature(self, uid, eventid):
        ''' See evernote  '''
        pass


    def loadFriendship(self, infile):
        user_friends = defaultdict(list)
        for entry in csv.reader(open(infile)):
            uid = entry[0]
            friends = entry[1].split(" ")
            user_friends[uid] = friends
        return user_friends

    def loadUserAttendEvent(self, infile):
        user_attend_event = defaultdict(list)
        for entry in csv.reader(open(infile)):
            uid = entry[0]
            eventids = entry[1].split(" ")
            user_attend_event[uid] = eventids
        return user_attend_event

    def loadEventInfo(self, infile):
        self.eventtype_id = {}
        self.words_id = {}
        term_num = defaultdict(int)
        for line in open(infile):
            entry = line.strip("\r\t\n").split(",")
            eventtype = entry[6]
            if eventtype not in self.eventtype_id:
                self.eventtype_id[eventtype] = len(self.eventtype_id)
            intro = entry[10]
            for term in set(intro.split(" ")):
                term = term.decode("utf8")
                tag = True
                for j in xrange(len(term)):
                    if not isChinese(term[j]):
                        tag = False
                if not tag:
                    continue
                term = term.encode("utf8")
                if term in self.stopwords:
                    continue
                term_num[term] += 1
        term_num = sorted(term_num.items(), key=lambda x:x[1], reverse=True)
        for pair in term_num[:settings["MAX_WORDS"]]:
            self.words_id[pair[0]] = len(self.words_id)

        self.word_idf = [0 for i in xrange(settings["MAX_WORDS"])]
        event_info = defaultdict(list)
        for line in open(infile):
            entry = line.strip("\r\t\n").split(",")
            eventid = entry[0]
            typeid = self.eventtype_id[entry[6]]
            entity_num = len(entry[11].split(" "))
            title_length = len(entry[4].split(" "))
            organizor = entry[5]
            start_time = entry[7]
            end_time = entry[8]
            temporal_period = getIdOfTimePeriod(start_time, end_time)
            intro_length = int(entry[9])
            location = entry[2]
            intro = []
            for word in entry[10].split(" "):
                if word in self.words_id:
                    intro.append(self.words_id[word])
            for word in set(intro):
                self.word_idf[self.words_id[word]] += 1
            event_info[eventid] = [typeid, intro_length, entity_num, title_length,
                    temporal_period, organizor, location, " ".join(intro)]
        total_event = 1.*len(event_info)
        for eventid in event_info:
            intro = event_info[eventid][6]
            tf_idf = [0.0 for i in xrange(settings["MAX_WORDS"])]
            intro_length = event_info[eventid][1]
            for word in intro.split(" "):
                tf_idf[word] += 1.0/intro_length*math.log(total_event/self.word_idf[word])
            event_info[eventid].append(tf_idf)
        return event_info

    def calOrganizorHoldEvent(self):
        organizor_hold_event = defaultdict(list)
        for eventid in self.event_info:
            organizor = self.event_info[eventid][5]
            organizor_hold_event[organizor].append(eventid)
        return organizor_hold_event

    def calUserContent(self):
        user_attend_content = {}
        for uid in self.user_attend_event:
            user_attend_content[uid] = np.array([0.0 for i in xrange(settings["MAX_WORDS"])])
            for eventid in self.user_attend_event[uid]:
                user_attend_content[uid] += self.event_info[eventid][-1]
            user_attend_content[uid] /= len(self.user_attend_event[uid])
        return user_attend_content

    def calUserCategoryPref(self):
        user_category_pref = {}
        for uid in self.user_attend_event:
            user_category_pref[uid] = np.array([0.0 for i in xrange(settings["CATEGORY_NUM"])])
            for eventid in self.user_attend_event[uid]:
                user_category_pref[uid][self.event_info[eventid][0]] += 1
            user_category_pref[uid] /= len(self.user_attend_event[uid])
        return user_category_pref

    def calUserLocation(self):
        user_location_addr = defaultdict(list)
        for uid in self.user_attend_event:
            for eventid in self.user_attend_event[uid]:
                location = self.event_info[eventid][6]
                user_location_addr[uid].append(location.split(" "))
        return user_location_addr

