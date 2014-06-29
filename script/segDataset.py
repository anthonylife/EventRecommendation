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
# Divide the data set into training, validation and test set.     #
###################################################################


import sys, csv, json, argparse, datetime, random
from collections import defaultdict

with open("../SETTINGS.json") as fp:
    settings = json.loads(fp.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, action='store',
            dest='city_num', help='choose the city whose data set will be used')
    parser.add_argument('-m', type=int, action='store',
            dest='seg_method', help='choose which method to divide the datset')

    if len(sys.argv) != 5:
        print 'Command e.g.: python segDataset.py -d 0(1) -m 0(1)'
        sys.exit(1)

    para = parser.parse_args()
    if para.city_num == 0:
        eventinfo_path = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_CITY1"]
        uae_path = settings["ROOT_PATH"] + settings["USER_EVENT_DATA1_CITY1"]
        out_train_path = settings["ROOT_PATH"] + settings["DATA1_CITY1_TRAIN"]
        out_vali_path = settings["ROOT_PATH"] + settings["DATA1_CITY1_VALIDATION"]
        out_test_path = settings["ROOT_PATH"] + settings["DATA1_CITY1_TEST"]
    elif para.city_num == 1:
        eventinfo_path = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_CITY2"]
        uae_path = settings["ROOT_PATH"] + settings["USER_EVENT_DATA1_CITY2"]
        out_train_path = settings["ROOT_PATH"] + settings["DATA1_CITY2_TRAIN"]
        out_vali_path = settings["ROOT_PATH"] + settings["DATA1_CITY2_VALIDATION"]
        out_test_path = settings["ROOT_PATH"] + settings["DATA1_CITY2_TEST"]
    else:
        print 'Invalid choice of city num'
        sys.exit(1)

    # (1) Get the number of events each user organized
    organizor_cnt = defaultdict(int)
    for line in open(eventinfo_path):
        parts = line.strip("\r\t\n").split(",")
        geoinfo = parts[3]
        if geoinfo == "0.0 0.0":
            continue
        organizor_cnt[parts[5]] += 1
    valid_organizor = set()
    for organizor in organizor_cnt:
        if organizor_cnt[organizor] >= settings["MIN_USER_ORGANIZE"]:
            valid_organizor.add(organizor)

    # (2) Get event time info and filter events without locations
    event_time = {}
    event_organizor = {}
    for line in open(eventinfo_path):
        parts = line.strip("\r\t\n").split(",")
        eventid = parts[0]
        geoinfo = parts[3]
        if geoinfo == "0.0 0.0":
            continue
        organizor = parts[5]
        if organizor not in valid_organizor:
            continue
        event_organizor[eventid] = organizor
        start_time = parts[7]
        event_time[eventid] = start_time

    # (3) Get events each user attended,
    user_events = defaultdict(list)
    for entry in csv.reader(open(uae_path)):
        uid = entry[0]
        eventids = entry[1].split(" ")
        for eventid in eventids:
            if eventid in event_time:
                user_events[uid].append([eventid, event_time[eventid],
                    event_organizor[eventid]])

    # (3) filtering users attending less than min support,
    #     and dividing the data set according to the time info.
    num_user = 0
    num_events = set([])
    num_organizor  = set([])
    writer_train = csv.writer(open(out_train_path, "w"), lineterminator="\n")
    writer_vali = csv.writer(open(out_vali_path, "w"), lineterminator="\n")
    writer_test = csv.writer(open(out_test_path, "w"), lineterminator="\n")
    for uid in user_events:
        if user_events[uid] >= settings["MIN_USER_ATTEND"]:
            num_user += 1
            if para.seg_method == 0:
                entries = sorted(user_events[uid], key=lambda x:x[1], reverse=False)
            elif para.seg_method == 1:
                random.shuffle(user_events[uid])
            else:
                print 'Invalid choice of segmentation method.'
                sys.exit(1)
            len_event = len(entries)
            start_idx = 0
            end_idx = int(len_event*settings["TRAIN_RATIO"])
            for idx in xrange(start_idx, end_idx):
                writer_train.writerow([uid]+user_events[uid][idx])
                num_events.add(user_events[uid][idx][0])
                num_organizor.add(user_events[uid][idx][2])
            start_idx = end_idx
            end_idx = start_idx + int(len_event*settings["VALI_RATIO"])
            for idx in xrange(start_idx, end_idx):
                writer_vali.writerow([uid]+user_events[uid][idx])
                num_events.add(user_events[uid][idx][0])
                num_organizor.add(user_events[uid][idx][2])
            start_idx = end_idx
            for idx in xrange(start_idx, len(user_events[uid])):
                writer_test.writerow([uid]+user_events[uid][idx])
                num_events.add(user_events[uid][idx][0])
                num_organizor.add(user_events[uid][idx][2])
    print "Basic statistics information:"
    print "\tNumber of users: %d" % num_user
    print "\tNumber of events: %d" % len(num_events)
    print "\tNumber of organizors: %d" % len(num_organizor)

if __name__ == "__main__":
    main()

