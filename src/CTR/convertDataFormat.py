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
# Date: 2014/7/1                                                  #
# Converting our raw specified data format to satisfy the input   #
#   requirements of CTR model.                                    #
###################################################################

import sys, csv, json, argparse, random
sys.path.append("../")
from collections import defaultdict
from utils import isChinese, getStopwords

settings = json.loads(open("../../SETTINGS.json").read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, action='store',
            dest='data_num', help='choose which data set to use')

    if len(sys.argv) != 3:
        print 'Command e.g.: python convertDataFormat.py -d 0(1)'
        sys.exit(1)

    para = parser.parse_args()
    if para.data_num == 0:
        event_info_path = settings["ROOT_PATH"]+settings["SRC_DATA_FILE1_CITY1"]
        tr_user_event_path = settings["ROOT_PATH"]+settings["DATA1_CITY1_TRAIN"]
        va_user_event_path = settings["ROOT_PATH"]+settings["DATA1_CITY1_VALIDATION"]
        te_user_event_path = settings["ROOT_PATH"]+settings["DATA1_CITY1_TEST"]
        user_records_path = settings["ROOT_PATH"]+settings["DATA1_CITY1_USER_RECORDS_CTR"]
        event_records_path = settings["ROOT_PATH"]+settings["DATA1_CITY1_EVENT_RECORDS_CTR"]
        event_intro_path = settings["ROOT_PATH"]+settings["DATA1_CITY1_EVENT_INFO_CTR"]
    elif para.data_num == 1:
        event_info_path = settings["ROOT_PATH"]+settings["SRC_DATA_FILE1_CITY2"]
        tr_user_event_path = settings["ROOT_PATH"]+settings["DATA1_CITY2_TRAIN"]
        va_user_event_path = settings["ROOT_PATH"]+settings["DATA1_CITY2_VALIDATION"]
        te_user_event_path = settings["ROOT_PATH"]+settings["DATA1_CITY2_TEST"]
        user_records_path = settings["ROOT_PATH"]+settings["DATA1_CITY2_USER_RECORDS_CTR"]
        event_records_path = settings["ROOT_PATH"]+settings["DATA1_CITY2_EVENT_RECORDS_CTR"]
        event_intro_path = settings["ROOT_PATH"]+settings["DATA1_CITY2_EVENT_INFO_CTR"]
    else:
        print 'Invalid choice of dataset'
        sys.exit(1)

    stopwords = getStopwords()

    ## output mapping id files
    user_ids = {}
    event_ids = {}
    for entry in csv.reader(open(tr_user_event_path)):
        uid = entry[0]
        eventid = entry[1]
        if uid not in user_ids:
            user_ids[uid] = len(user_ids)
        if eventid not in event_ids:
            event_ids[eventid] = len(event_ids)
    for entry in csv.reader(open(va_user_event_path)):
        eventid = entry[1]
        if eventid not in event_ids:
            event_ids[eventid] = len(event_ids)
    for entry in csv.reader(open(te_user_event_path)):
        eventid = entry[1]
        if eventid not in event_ids:
            event_ids[eventid] = len(event_ids)
    writer1 = csv.writer(open(settings["ROOT_PATH"]+settings["USER_IDS_PATH"], "w"), lineterminator="\n")
    writer2 = csv.writer(open(settings["ROOT_PATH"]+settings["EVENT_IDS_PATH"], "w"), lineterminator="\n")
    for uid in user_ids:
        writer1.writerow([uid, user_ids[uid]])
    for eventid in event_ids:
        writer2.writerow([eventid, event_ids[eventid]])

    ## output user and event records files
    event_user = [[] for i in xrange(len(event_ids))]
    user_event = [[] for i in xrange(len(user_ids))]
    for entry in csv.reader(open(tr_user_event_path)):
        uid = user_ids[entry[0]]
        eventid = event_ids[entry[1]]
        event_user[eventid].append(uid)
        user_event[uid].append(eventid)
    wfd = open(user_records_path, "w")
    for events in user_event:
        wfd.write("%d" % len(events))
        for event in events:
            wfd.write(" %d" % event)
        wfd.write("\n")
    wfd.close()
    wfd = open(event_records_path, "w")
    for users in event_user:
        wfd.write("%d" % len(users))
        for user in users:
            wfd.write(" %d" % user)
        wfd.write("\n")
    wfd.close()

    ## output event introduction
    words_id = {}
    term_num = defaultdict(int)
    for line in open(event_info_path):
        entry = line.strip("\r\t\n").split(",")
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
            if term in stopwords:
                continue
            term_num[term] += 1
    term_num = sorted(term_num.items(), key=lambda x:x[1], reverse=True)
    for pair in term_num[:settings["MAX_WORDS"]]:
        words_id[pair[0]] = len(words_id)

    event_intro = [[] for i in xrange(len(event_ids))]
    for line in open(event_info_path):
        entry = line.strip("\r\t\n").split(",")
        if entry[0] not in event_ids:
            continue
        eventid = event_ids[entry[0]]
        intro = entry[10]
        for term in set(intro.split(" ")):
            if term in words_id:
                event_intro[eventid].append(words_id[term])

    wfd = open(event_intro_path, "w")
    for intro in event_intro:
        term_dict = defaultdict(int)
        for term in intro:
            term_dict[term] += 1
        wfd.write("%d" % len(term_dict))
        for term in term_dict:
            wfd.write(" %d:%d" % (term, term_dict[term]))
        wfd.write("\n")
        del term_dict
    wfd.close()

if __name__ == "__main__":
    main()

