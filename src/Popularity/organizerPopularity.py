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
# Date: 2014/7/2                                                  #
# Recommending events only based on the popularity level of       #
#   organizers.                                                   #
###################################################################


import sys, csv, json, argparse, random
sys.path.append("../")
from collections import defaultdict

settings = json.loads(open("../../SETTINGS.json").read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, action='store',
            dest='data_num', help='choose which data set to use')

    if len(sys.argv) != 3:
        print 'Command e.g.: python organizerPopularity -d 0(1)'
        sys.exit(1)

    para = parser.parse_args()
    if para.data_num == 0:
        event_intro_path = settings["ROOT_PATH"]+settings["SRC_DATA_FILE1_CITY1"]
        event_train_path = settings["ROOT_PATH"]+settings["DATA1_CITY1_TRAIN"]
        event_test_path = settings["ROOT_PATH"]+settings["DATA1_CITY1_TEST"]
        result_path = settings["ROOT_PATH"]+settings["POPULARITY_RESULT1"]
    elif para.data_num == 1:
        event_intro_path = settings["ROOT_PATH"]+settings["SRC_DATA_FILE1_CITY2"]
        event_train_path = settings["ROOT_PATH"]+settings["DATA1_CITY2_TRAIN"]
        event_test_path = settings["ROOT_PATH"]+settings["DATA1_CITY2_TEST"]
        result_path = settings["ROOT_PATH"]+settings["POPULARITY_RESULT2"]
    else:
        print 'Invalid choice of dataset'
        sys.exit(1)

    train_events = set([])
    for entry in csv.reader(open(event_train_path)):
        eventid = entry[1]
        train_events.add(eventid)

    event_organizer = {}
    organizer_eventlist = defaultdict(list)
    for line in open(event_intro_path):
        parts = line.strip("\r\t\n").split(" ")
        eventid = parts[0]
        if eventid not in train_events:
            continue
        organizer = parts[5]
        usernum = int(parts[9])
        organizer_eventlist[organizer].append(usernum)
        event_organizer[eventid] = organizer

    organizer_popularity = {}
    for organizer in organizer_eventlist:
        organizer_popularity[organizer]=1.*sum(organizer_eventlist[organizer])/len(organizer_eventlist[organizer])

    candidate_users = set([])
    future_events = set([])
    for entry in csv.reader(open(event_test_path)):
        uid = entry[0]
        eventid = entry[1]
        candidate_users.add(uid)
        future_events.add(eventid)
    future_events = list(future_events)
    print "Basic statistics of testing"
    print "\tNumber of users %d, number of events %d!" % (len(candidate_users), len(future_events))

    ## Personalized Recommendation
    writer = csv.writer(open(result_path, "w"), lineterminator="\n")
    for uid in candidate_users:
        event_predictions = []
        for candidate_eventid in future_events:
            organizer = event_organizer[candidate_eventid]
            score = organizer_popularity[organizer]
            event_predictions.append([candidate_eventid, score])
        event_predictions = sorted(event_predictions, key=lambda x:x[1], reverse=True)
        prediction_result = [pair[0] for pair in event_predictions]
        writer.writerow([uid]+prediction_result)


if __name__=="__main__":
    main()
