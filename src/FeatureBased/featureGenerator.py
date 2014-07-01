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


import sys, csv, json, math
sys.path.append("../")
import numpy as np
from geopy import distance
from collections import defaultdict
from utils import getIdOfTimePeriod, isChinese, getStopwords

distance.distance = distance.GreatCircleDistance
settings = json.loads(open("../../SETTINGS.json").read())


class FeatureGenerator():
    def __init__(self, gen_feature_method, friend_path, event_path, tr_data_path):
        self.gen_feature_method = gen_feature_method
        self.stopwords = getStopwords()
        self.user_friendship = self.loadFriendship(friend_path)
        self.event_info = self.loadEventInfo(event_path)
        self.user_attend_event = self.loadUserAttendEvent(tr_data_path)
        self.organizor_hold_event = self.calOrganizorHoldEvent()
        self.user_attend_content = self.calUserContent()
        self.user_attend_category = self.calUserEventTypePref()
        self.user_attend_time = self.calUserEventTimePref()

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
            5.number of events organizor holds
            6.average number of users attending the above events
            7.temporal period of event
        '''
        if eventid not in self.event_info:
            print 'Event basic info loss!'
            sys.exit(1)
        feature = []
        eventtype = self.event_info[eventid][0]
        feature.append(eventtype)
        intro_length = self.event_info[eventid][1]
        feature.append(intro_length)
        entity_num = self.event_info[eventid][2]
        feature.append(entity_num)
        title_length = self.event_info[eventid][3]
        feature.append(title_length)
        organizor = self.event_info[eventid][5]
        event_num = self.organizor_hold_event[organizor][-1][0]
        average_user_num = self.organizor_hold_event[organizor][-1][1]
        feature.append(event_num)
        feature.append(average_user_num)
        temporal_period = self.event_info[eventid][4]
        feature.append(temporal_period)
        return feature

    def genBigramFeature(self, uid, eventid):
        ''' Bigram feature list:
            1.number of event user attending this event type;
            2.event intro similarity;
            3.number of events organized by the current organizer
              the target user attending;
            4.number of events organized by the current organizer
              the target user's friends attending (Note:Currently Missing);
            5.distance between user's event region and target event;
            6.user's active level in the corresponding temporal period.
        '''
        feature = []
        eventtype = self.event_info[eventid][0]
        event_num = self.calEventNumberForType(uid, eventid, eventtype)
        feature.append(event_num)
        intro_sim = self.calEventIntroSim(uid, eventid)
        feature.append(intro_sim)
        event_num = self.calAttendingEventForOrganizer(uid, self.event_info[eventid][5])
        feature.append(event_num)
        #event_num = self.calFriendAttendingEventForOrganizer(uid, self.event_info[eventid][5])
        #feature.append(event_num)
        distance = self.calAverageDistance(uid, eventid)
        feature.append(distance)
        active_level = self.calUserActiveLevelForTime(uid, eventid)
        feature.append(active_level)
        return feature

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
            eventid = entry[1]
            user_attend_event[uid].append(eventid)
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
            attend_user_num = int(entry[9])
            location = entry[3]
            intro = []
            for word in entry[10].split(" "):
                if word in self.words_id:
                    intro.append(self.words_id[word])
            intro_length = len(intro)
            for wordid in set(intro):
                self.word_idf[wordid] += 1
            event_info[eventid] = [typeid, intro_length, entity_num, title_length,
                    temporal_period, organizor, location, attend_user_num, intro]
        total_event = 1.*len(event_info)

        for eventid in event_info:
            intro = event_info[eventid][-1]
            tf_idf = [0.0 for i in xrange(settings["MAX_WORDS"])]
            intro_length = event_info[eventid][1]
            for wordid in intro:
                tf_idf[wordid] += 1.0/intro_length*math.log(total_event/self.word_idf[wordid])
            event_info[eventid].append(tf_idf)
        return event_info

    def calOrganizorHoldEvent(self):
        organizor_hold_event = defaultdict(list)
        # constraining user attending records only from training data
        for uid in self.user_attend_event:
            for eventid in self.user_attend_event[uid]:
                organizor = self.event_info[eventid][5]
                organizor_hold_event[organizor].append([eventid, self.event_info[eventid][7]])
        for organizor in organizor_hold_event:
            event_num = len(organizor_hold_event[organizor])
            average_user_num = 1.*sum([pair[1] for pair in organizor_hold_event[organizor]])/event_num
            organizor_hold_event[organizor].append([event_num, average_user_num])
        return organizor_hold_event

    def calUserContent(self):
        user_attend_content = {}
        for uid in self.user_attend_event:
            user_attend_content[uid] = np.array([0.0 for i in xrange(settings["MAX_WORDS"])])
            for eventid in self.user_attend_event[uid]:
                user_attend_content[uid] += self.event_info[eventid][-1]
            user_attend_content[uid] /= len(self.user_attend_event[uid])
        return user_attend_content

    def calUserEventTypePref(self):
        user_category_pref = {}
        for uid in self.user_attend_event:
            user_category_pref[uid] = np.array([0.0 for i in xrange(settings["CATEGORY_NUM"])])
            for eventid in self.user_attend_event[uid]:
                user_category_pref[uid][self.event_info[eventid][0]] += 1
            user_category_pref[uid] /= len(self.user_attend_event[uid])
        return user_category_pref

    def calUserEventTimePref(self):
        user_time_pref = {}
        for uid in self.user_attend_event:
            user_time_pref[uid] = np.array([0.0 for i in xrange(settings["PERIOD_NUM"])])
            for eventid in self.user_attend_event[uid]:
                user_time_pref[uid][self.event_info[eventid][4]] += 1
            user_time_pref[uid] /= len(self.user_attend_event[uid])
        return user_time_pref

    def calUserLocation(self):
        user_location_addr = defaultdict(list)
        for uid in self.user_attend_event:
            for eventid in self.user_attend_event[uid]:
                location = self.event_info[eventid][6]
                user_location_addr[uid].append(location.split(" "))
        return user_location_addr

    def calEventNumberForType(self, quid, qeventid, qeventtype):
        event_num = self.user_attend_category[quid][qeventtype]*len(self.user_attend_event[quid])
        if self.gen_feature_method == 0:
            return event_num-1
        else:
            return event_num

    def calEventIntroSim(self, quid, qeventid):
        if self.gen_feature_method == 0:
            user_intro = self.user_attend_content[quid]*len(self.user_attend_event[quid])
            user_intro -= self.event_info[qeventid][-1]
            if (len(self.user_attend_event[quid])-1)==0:
                print 'Unexpected error occur: user attending so few events.'
                sys.exit(1)
            user_intro /= len(self.user_attend_event[quid])-1
        else:
            user_intro = self.user_attend_content[quid]
        norm1 = np.linalg.norm(user_intro)
        norm2 = np.linalg.norm(self.event_info[qeventid][-1])
        if norm1 != 0 and norm2 != 0:
            return np.dot(user_intro, self.event_info[qeventid][-1])/np.linalg.norm(user_intro)/np.linalg.norm(self.event_info[qeventid][-1])
        else:
            return 0.0

    def calAttendingEventForOrganizer(self, quid, qorganizer):
        event_num = 0
        for eventid in self.user_attend_event[quid]:
            if self.event_info[eventid][5] == qorganizer:
                event_num += 1
        if self.gen_feature_method == 0:
            return event_num-1
        else:
            return event_num

    def calAverageDistance(self, quid, qeventid):
        average_distance = 0.0
        lat_q, lng_q = self.event_info[qeventid][6].split(" ")
        lat_q = float(lat_q)
        lng_q = float(lng_q)
        for eventid in self.user_attend_event[quid]:
            lat, lng = self.event_info[qeventid][6].split(" ")
            lat = float(lat)
            lng = float(lng)
            mile = distance.distance((lat_q, lng_q), (lat, lng)).miles
            average_distance += mile
        if self.gen_feature_method == 0:
            return average_distance/(len(self.user_attend_event[quid])-1)
        else:
            return average_distance/len(self.user_attend_event[quid])

    def calUserActiveLevelForTime(self, quid, qeventid):
        time_period = self.event_info[qeventid][4]
        if self.gen_feature_method == 0:
            return (self.user_attend_time[quid][time_period]*len(self.user_attend_event[quid])-1)/(len(self.user_attend_event[quid])-1)
        else:
            return self.user_attend_time[quid][time_period]
