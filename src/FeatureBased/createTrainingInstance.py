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
# Date: 2014/6/30                                                 #
# Adopting sub sampling strategy to generate pairs (uid, eventid) #
# with positive and negative lables.                              #
###################################################################


import sys, csv, json, argparse, random, math
sys.path.append("../")
import numpy as np
from collections import defaultdict
from featureGenarator import FeatureGenerator

settings = json.loads(open("../../SETTINGS.json").read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, action='store',
            dest='data_num', help='choose which data set to use')

    if len(sys.argv) != 3:
        print 'Command e.g.: python createTrainingInstance.py -d 1(2)'
        sys.exit(1)

    para = parser.parse_args()
    if para.data_num == 1:
        event_intro_path = settings["ROOT_PATH"]+settings["SRC_DATA_FILE1_CITY1"]
        event_train_path = settings["ROOT_PATH"]+settings["DATA1_CITY1_TRAIN"]
        user_friend_path = settings["ROOT_PATH"]+settings["SRC_DATA_FILE1_3"]
        out_feature_path = settings["ROOT_PATH"]+settings["DATA1_CITY1_TRAIN_FEATURE"]
    elif para.data_num == 2:
        event_intro_path = settings["ROOT_PATH"]+settings["SRC_DATA_FILE1_CITY2"]
        event_train_path = settings["ROOT_PATH"]+settings["DATA1_CITY2_TRAIN"]
        user_friend_path = settings["ROOT_PATH"]+settings["SRC_DATA_FILE1_3"]
        out_feature_path = settings["ROOT_PATH"]+settings["DATA1_CITY2_TRAIN_FEATURE"]
    else:
        print 'Invalid choice of dataset'
        sys.exit(1)

    featureGenarator = FeatureGenerator(user_friend_path, event_intro_path,
            event_train_path)

    event_set = set([])
    user_pos_event = defaultdict(list)
    for entry in csv.reader(open(event_train_path)):
        uid = entry[0]
        eventid = entry[1]
        event_set.add(eventid)
        user_pos_event[uid].append(eventid)

    writer = csv.writer(open(out_feature_path, "w"), lineterminator="\n")
    for uid in user_pos_event:
        for eventid in user_pos_event[uid]:
            feature = featureGenarator.genFeature(uid, eventid)
            writer.writerow(feature)
            neg_eventids = random.sample(event_set-set(user_pos_event[uid]), settings["SAMPLE_RATIO"])
            for neg_eventid in neg_eventids:
                feature = featureGenarator.genFeature(uid, eventid)
                writer.writerow(feature)

if __name__=="__main__":
    main()

