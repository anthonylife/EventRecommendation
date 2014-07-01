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
# Evaluation on event recommendation results                      #
# Note:                                                           #
#   1. evaluation metrics including P@k, MAP                      #
###################################################################


import sys, csv, json, argparse, math
from collections import defaultdict

with open("../SETTINGS.json") as fp:
    settings = json.loads(fp.read())


class Evaluation():
    def __init__(self, standard_result_file, prediction_result_file):
        self.standard_result_file = standard_result_file
        self.prediction_result_file = prediction_result_file
        self.user_standard_result = self.loadStandardResult()
        #self.user_prediction_result = self.loadPredictionResult()

    def loadStandardResult(self):
        self.user_standard_result = defaultdict(set)
        for entry in csv.reader(open(self.standard_result_file)):
            uid = entry[0]
            eventid = entry[1]
            self.user_standard_result[uid].add(eventid)

    def loadPredictionResult(self):
        self.user_prediction_result = defaultdict(list)
        for entry in csv.reader(open(self.prediction_result_file)):
            uid = entry[0]
            for eventid in entry[1:]:
                self.user_prediction_result[uid].append(eventid)

    def evaluate(self, eval_method=0, topk=0):
        if eval_method == 0:
            print 'MAP evaluation!'
            eval_map = 0.0
            valid_user_num = 0
            for entry in csv.reader(open(self.prediction_result_file)):
                uid = entry[0]
                if uid not in self.user_standard_result:
                    continue
                valid_user_num += 1
                standard_num = len(self.user_standard_result[uid])
                eval_ap = 0.0
                correct_num = 0
                for idx, eventid in enumerate(entry[1:]):
                    if eventid in self.user_standard_result[uid]:
                        correct_num += 1
                        eval_ap += 1.*correct_num/(idx+1)
                        if correct_num == standard_num:
                            break
                eval_ap /= standard_num
                eval_map += eval_ap
            return eval_map/valid_user_num
        elif eval_method == 1:
            print 'P@%d evaluation!' % topk
            eval_prec = 0.0
            valid_user_num = 0
            for entry in csv.reader(open(self.prediction_result_file)):
                uid = entry[0]
                if uid not in self.user_standard_result:
                    continue
                valid_user_num += 1
                correct_num = 0
                for eventid in entry[1:topk+1]:
                    if eventid in self.user_standard_result[uid]:
                        correct_num += 1
                eval_prec += 1.*correct_num/topk
            return eval_prec/valid_user_num
        else:
            print 'Invalid choice of evaluation method!'
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, action='store',
            dest='data_num', help='choose which data set to use')
    parser.add_argument('-a', type=int, action='store',
            dest='algorithm_num', help='specify the algorithm which genenrate the recommendation results')
    if len(sys.argv) != 5:
        print 'Command e.g.: python evaluation.py -d 0(1) -a 0(0,1,...)'
        sys.exit(1)

    para = parser.parse_args()
    if para.data_num == 0:
        standard_result_file = settings["ROOT_PATH"]+settings["DATA1_CITY1_TEST"]
        if para.algorithm_num == 0:
            prediction_result_file = settings["ROOT_PATH"]+settings["POPULARITY_RESULT1"]
            tips = "Douban Beijing data-->Organizer based propualrity algorithm"
        elif para.algorithm_num == 1:
            prediction_result_file = settings["ROOT_PATH"]+settings["FEATURE_MODEL_RESULT1"]
            tips = "Douban Beijing data-->Feature based algorithm"
        else:
            print 'Invalid choice of algorithm!'
            sys.exit(1)
    elif para.data_num == 1:
        standard_result_file = settings["ROOT_PATH"]+settings["DATA1_CITY2_TEST"]
        if para.algorithm_num == 0:
            prediction_result_file = settings["ROOT_PATH"]+settings["POPULARITY_RESULT2"]
            tips = "Douban Shanghai data-->Organizer based propualrity algorithm"
        elif para.algorithm_num == 1:
            prediction_result_file = settings["ROOT_PATH"]+settings["FEATURE_MODEL_RESULT2"]
            tips = "Douban Shanghai data-->Feature based algorithm"
        else:
            print 'Invalid choice of algorithm!'
            sys.exit(1)
    else:
        print 'Invalid choice of data set!'
        sys.exit(1)

    evaluation = Evaluation(standard_result_file,
                            prediction_result_file)
    eval_map = evaluation.evaluate(0)
    print "%s: MAP=%.3f!" %(tips, eval_map)

if __name__ == "__main__":
    main()

