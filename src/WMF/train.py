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
# Call model.py to do weighted matrix factorization               #
#   for implicit feedback.                                        #
###################################################################

import sys, csv, json, argparse
sys.path.append("../")
from model import WMF

settings = json.loads(open("../../SETTINGS.json").read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-init', type=str, action='store', dest='init_choice',
            help='specify which method to initialize model parameters')
    parser.add_argument('-r', type=str, action='store',dest='retrain_choice',
            help='specify which method to initialize model parameters')
    parser.add_argument('-d', type=int, action='store',
            dest='data_num', help='choose which data set to use')

    if len(sys.argv) != 7:
        print 'Command e.g.: python train.py -retrain True -init zero(gaussian)'
        sys.exit(1)

    para = parser.parse_args()
    wmf = WMF()
    if para.data_num == 0:
        event_info_path = settings["ROOT_PATH"]+settings["SRC_DATA_FILE1_CITY1"]
        event_train_path = settings["ROOT_PATH"]+settings["DATA1_CITY1_TRAIN"]
        event_test_path = settings["ROOT_PATH"]+settings["DATA1_CITY1_TEST"]
        result_path = settings["ROOT_PATH"]+settings["WMF_RESULT1"]
    elif para.data_num == 1:
        event_info_path = settings["ROOT_PATH"]+settings["SRC_DATA_FILE1_CITY2"]
        event_train_path = settings["ROOT_PATH"]+settings["DATA1_CITY2_TRAIN"]
        event_test_path = settings["ROOT_PATH"]+settings["DATA1_CITY2_TEST"]
        result_path = settings["ROOT_PATH"]+settings["WMF_RESULT2"]
    else:
        print 'Invalid choice of dataset'
        sys.exit(1)

    if para.retrain_choice == "True":
        wmf.model_init(event_train_path, para.init_choice)
        wmf.train()
        wmf.genRecommendResult(True, event_test_path, para.init_choice, result_path)
    else:
        wmf.genRecommendResult(False, event_test_path, para.init_choice, result_path)

if __name__ == "__main__":
    main()
