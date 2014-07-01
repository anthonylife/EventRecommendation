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
# Date: 2014/6/17                                                 #
# Providing useful functions.                                     #
###################################################################

import sys, csv, json, argparse, datetime, pickle
from collections import defaultdict

with open("../../SETTINGS.json") as fp:
    settings = json.loads(fp.read())
dt = datetime.datetime.now()


def getIdOfTimePeriod(start_time, end_time):
    time1 = dt.strptime(start_time, '%Y-%m-%dT%H:%M:%S+08:00')
    time2 = dt.strptime(end_time, '%Y-%m-%dT%H:%M:%S+08:00')
    year1 = time1.year
    year2 = time2.year
    #day1 = time1.day
    #day2 = time2.day
    day1 = int(time1.strftime('%j'))
    day2 = int(time2.strftime('%j'))
    if year1 == year2:
        if day2-day1 > 30:
            return 10
        elif day2-day1 > 7:
            return 9
        elif day2-day1 > 0:
            return 8
        elif day2 == day1:
            idx1= 0
            idx2= 0
            hour1 = time1.hour
            #hour2 = time2.hour
            weekday1 = time1.isoweekday()
            if weekday1 == 6 or weekday1 == 7:
                idx1 = 1
            if 8 <= hour1 and hour1 < 12:
                idx2 = 0
            elif 12 <= hour1 and hour1 < 18:
                idx2 = 1
            elif 18 <= hour1 and hour1 < 24:
                idx2 = 2
            else:
                idx2 = 3
            return idx1*4+idx2

    elif year1+1 == year2:
        if day2+366-day1 > 30:
            return 10
        elif day2+366-day1 > 7:
            return 9
        elif day2+366-day1 > 0:
            return 8
        else:
            print 'Error in getting id of time period'
            sys.exit(1)
    else:
        return 10

def isChinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False

def getStopwords():
    stopwords = set([])
    for line in open(settings["ROOT_PATH"]+settings["STOP_WORD_FILE"]):
        term = line.strip("\r\t\n")
        stopwords.add(term)
    return stopwords

def write_submission(prediction_result, result_file):
    writer = csv.writer(open(result_file, "w"))
    id_sorted_result = sorted(prediction_result.items(), key=lambda x: x[0])
    for pair in id_sorted_result:
        writer.writerow(pair)

def save_model(model, out_path):
    pickle.dump(model, open(out_path, "w"))

def load_model(in_path):
    return pickle.load(open(in_path))

