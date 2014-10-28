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
# Date: 2014/10/21                                                #
# Call model.py to do GeoMF for Event Recommendation.             #
###################################################################

import sys, csv, json, argparse
sys.path.append("../../")
from model import GeoMF_D, GeoMF_O, HeSig, Distance

settings = json.loads(open("../../../SETTINGS.json").read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, action='store', dest='model_id',
            help='choose which model to learn from data')
    parser.add_argument('-d', type=int, action='store',
            dest='data_id', help='choose which data set to use')

    if len(sys.argv) != 5:
        print 'Command e.g.: python train.py -m 0(1,2,3,4) -d 0(1)'
        sys.exit(1)

    result_paths = [[settings["GEOMF-D_RESULT1"], settings["GEOMF-K_RESULT1"], settings["GEOMF-O_RESULT1"], settings["HESIG_RESULT1"], settings["DISTANCE_RESULT1"]], [settings["GEOMF-D_RESULT1"], settings["GEOMF-K_RESULT1"], settings["GEOMF-O_RESULT1"], settings["HESIG_RESULT1"], settings["DISTANCE_RESULT1"]]]

    para = parser.parse_args()
    if para.data_id == 0:
        event_info_path = settings["ROOT_PATH"]+settings["SRC_DATA_FILE1_CITY1"]
        event_train_path = settings["ROOT_PATH"]+settings["DATA1_CITY1_TRAIN"]
        event_test_path = settings["ROOT_PATH"]+settings["DATA1_CITY1_TEST"]
    elif para.data_id == 1:
        event_info_path = settings["ROOT_PATH"]+settings["SRC_DATA_FILE1_CITY2"]
        event_train_path = settings["ROOT_PATH"]+settings["DATA1_CITY2_TRAIN"]
        event_test_path = settings["ROOT_PATH"]+settings["DATA1_CITY2_TEST"]
    else:
        print 'Invalid choice of dataset'
        sys.exit(1)

    result_path = settings["ROOT_PATH"] + result_paths[para.data_id][para.model_id]

    if para.model_id == 0:
        model = GeoMF_D(1, True, event_train_path, event_info_path, para.data_id)
        model.model_init(event_train_path, event_info_path)
    elif para.model_id == 1:
        model = GeoMF_D(2, True, event_train_path, event_info_path, para.data_id)
        model.model_init(event_train_path, event_info_path)
    elif para.model_id == 2:
        model = GeoMF_O(para.data_id)
        model.model_init(event_train_path, event_info_path)
    elif para.model_id == 3:
        model = HeSig(2, True, event_train_path, event_info_path, para.data_id)
        model.model_init(event_train_path, event_info_path)
    elif para.model_id == 4:
        model = Distance(para.data_id)
        model.model_init(event_train_path, event_info_path)
    else:
        print 'Invalid choice of model'
        sys.exit(1)

    model.train()
    model.genRecommendResult(event_test_path, result_path)

if __name__ == "__main__":
    main()
