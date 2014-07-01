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
# Supervised learning for event recommendation                    #
###################################################################

import csv, json, sys, argparse
sys.path.append("../")
from utils import save_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

settings = json.loads(open("../../SETTINGS.json").read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, action='store',
            dest='data_num', help='choose which data set to use')
    parser.add_argument('-m', type=int, action='store',
            dest='method', help='choose which model to train')
    if len(sys.argv) != 5:
        print 'Command e.g.: python train.py -d 0(1) -m 0(1,...)'
        sys.exit(1)

    para = parser.parse_args()
    if para.data_num == 0:
        feature_file = settings["ROOT_PATH"] + settings["DATA1_CITY1_TRAIN_FEATURE"]
    elif para.data_num == 1:
        feature_file = settings["ROOT_PATH"] + settings["DATA1_CITY2_TRAIN_FEATURE"]
    else:
        print 'Invalid choice of dataset'
        sys.exit(1)
    model_path = settings["FEATURE_MODEL_PATH"]

    print("Loading data")
    features = []
    labels = []
    for i, entry in enumerate(csv.reader(open(feature_file))):
        label = int(entry[0])
        labels.append(label)
        feature = [float("%.2f" % float(val)) for val in entry[3:]]
        features.append(feature)
        if (i%1000) == 0 and i != 0:
            sys.stdout.write("\rFINISHED TRAINING NUM: %d. " % (i+1))
            sys.stdout.flush()

    print("Training the Model")
    if para.method == 0:
        classifier = LogisticRegression(penalty='l2',
                                        dual=False,
                                        tol=0.0001,
                                        C=10,
                                        fit_intercept=True,
                                        intercept_scaling=1,
                                        class_weight=None,
                                        random_state=None)
    elif para.method == 1:
        classifier = GradientBoostingClassifier(n_estimators=50,
                                                verbose=2,
                                                n_jobs=1,
                                                min_samples_split=10,
                                                random_state=1)
    elif para.method == 2:
        classifier = RandomForestClassifier(n_estimators=100,
                                            verbose=2,
                                            n_jobs=1,
                                            min_samples_split=10,
                                            random_state=1)
    else:
        print 'Invalid choice of model'
        sys.exit(1)

    classifier.fit(features, labels)
    print("Saving the classifier")
    save_model(classifier, model_path)


if __name__=="__main__":
    main()

