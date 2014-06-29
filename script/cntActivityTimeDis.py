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
# Date: 2014/6/15                                                 #
# Count distribution of number of activities for each time period.#
###################################################################


import sys, csv, json, argparse, datetime
from collections import defaultdict

with open("../SETTINGS.json") as fp:
    settings = json.loads(fp.read())
dt = datetime.datetime.now()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, action='store',
            dest='city_num', help='choose the city whose data set will be used')
    parser.add_argument('-t', type=int, action='store',
            dest='time_granularity', help='choose the time granularity')

    if len(sys.argv) != 5:
        print 'Command e.g.: python cntBasicStatics.py -d 0(1) -t 0(1)'
        sys.exit(1)

    para = parser.parse_args()
    if para.city_num == -1:
        eventinfo_path = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_1"]
    elif para.city_num == 0:
        eventinfo_path = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_CITY1"]
    elif para.city_num == 1:
        eventinfo_path = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_CITY2"]
    else:
        print 'Invalid choice of city num'
        sys.exit(1)

    stainfo_path = "./staresult.txt"

    time_num = defaultdict(int)
    if para.time_granularity == 0:
        for line in open(eventinfo_path):
            time_str = line.strip("\r\t\n").split(",")[8]
            time = dt.strptime(time_str, '%Y-%m-%dT%H:%M:%S+08:00')
            year, month, day = time.year, time.month, time.day
            week = time.strftime("%U")
            time_num["_".join([str(year), str(month), str(week), str(day)])] += 1

        wfd = open(stainfo_path, "w")
        time_num = sorted(time_num.items(), key=lambda x:x[0])
        for entry in time_num:
            year, month, week, day = entry[0].split("_")
            wfd.write("%s %s %s %s %d\n" % (year, month, week, day, entry[1]))
        wfd.close()
    elif para.time_granularity == 1:
        for line in open(eventinfo_path):
            time_str = line.strip("\r\t\n").split(",")[8]
            time = dt.strptime(time_str, '%Y-%m-%dT%H:%M:%S+08:00')
            year, month, day = time.year, time.month, time.day
            week = time.strftime("%U")
            time_num["_".join([str(year), str(month), str(week)])] += 1

        wfd = open(stainfo_path, "w")
        time_num = sorted(time_num.items(), key=lambda x:x[0])
        for entry in time_num:
            year, month, week = entry[0].split("_")
            wfd.write("%s %s %s %d\n" % (year, month, week, entry[1]))
        wfd.close()


if __name__ == "__main__":
    main()
