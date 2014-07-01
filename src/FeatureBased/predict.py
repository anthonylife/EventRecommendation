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
# Supervised learning for event recommendation                    #
###################################################################

import sys, csv, json, argparse
sys.path.append("../")
from collections import defaultdict
from utils import load_model, write_submission
from featureGenerator import FeatureGenerator
from utils import tic, toc

settings = json.loads(open("../../SETTINGS.json").read())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, action='store',
            dest='data_num', help='choose which data set to use')
    if len(sys.argv) != 3:
        print 'Command e.g.: python predict.py -d 0(1)'
        sys.exit(1)

    para = parser.parse_args()
    if para.data_num == 0:
        event_intro_path = settings["ROOT_PATH"]+settings["SRC_DATA_FILE1_CITY1"]
        event_train_path = settings["ROOT_PATH"]+settings["DATA1_CITY1_TRAIN"]
        user_friend_path = settings["ROOT_PATH"]+settings["SRC_DATA_FILE1_3"]
        event_test_path = settings["ROOT_PATH"] + settings["DATA1_CITY1_TEST"]
        result_path = settings["ROOT_PATH"] + settings["FEATURE_MODEL_RESULT1"]
    elif para.data_num == 2:
        event_intro_path = settings["ROOT_PATH"]+settings["SRC_DATA_FILE1_CITY2"]
        event_train_path = settings["ROOT_PATH"]+settings["DATA1_CITY2_TRAIN"]
        user_friend_path = settings["ROOT_PATH"]+settings["SRC_DATA_FILE1_3"]
        event_test_path = settings["ROOT_PATH"] + settings["DATA1_CITY2_TEST"]
        result_path = settings["ROOT_PATH"] + settings["FEATURE_MODEL_RESULT2"]
    else:
        print 'Invalid choice of dataset'
        sys.exit(1)


    print("Loading data")
    candidate_users = set([])
    future_events = set([])
    for entry in csv.reader(open(event_test_path)):
        uid = entry[0]
        eventid = entry[1]
        candidate_users.add(uid)
        future_events.add(eventid)
    future_events = list(future_events)

    print "Basic statistics of testing"
    print "\tNumber of users %d, number of events %d!" % (len(candidate_users),
            len(future_events))

    ## Personalized Recommendation
    writer = csv.writer(open(result_path, "w"), lineterminator="\n")
    #user_prediction_result = defaultdict(list)
    featureGenarator = FeatureGenerator(1, user_friend_path, event_intro_path,
            event_train_path)
    model_path = settings["FEATURE_MODEL_PATH"]
    classifier = load_model(model_path)
    finish_num = 0
    for uid in candidate_users:
        features = []
        for candidate_eventid in future_events:
            feature = featureGenarator.genFeature(uid, candidate_eventid)
            features.append(feature)
        predictions = classifier.predict_proba(features)[:,1]
        predictions = list(predictions)
        event_predictions = [[eventid, prediction] for eventid, prediction in zip(future_events, predictions)]
        event_predictions = sorted(event_predictions, key=lambda x:x[1], reverse=True)
        #user_prediction_result[uid] = [pair[0] for pair in event_predictions][:settings["RE_TOPK"]]
        prediction_result = [pair[0] for pair in event_predictions]
        writer.writerow([uid]+prediction_result)
        finish_num += 1
        if (finish_num%1000) == 0 and finish_num != 0:
            sys.stdout.write("\rFINISHED TRAINING NUM: %d. " % (finish_num+1))
            toc()
            sys.stdout.flush()
            tic()
    #write_submission(user_prediction_result, result_path)

if __name__=="__main__":
    main()
