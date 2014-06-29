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
# Date: 2014/5/26                                                 #
# Count the basic statistics of the specified dataset             #
#  e.g. 1.User number; 2.Event number;                            #
#       4.User average event num; 5. Event average user num;      #
#       6.User average event distribution.                        #
###################################################################

import sys, csv, json, argparse, pylab
import numpy as np
from collections import defaultdict

reload(sys)
sys.setdefaultencoding('utf8')

with open("../SETTINGS.json") as fp:
    settings = json.loads(fp.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, action='store',
            dest='data_num', help='choose which data set to use')

    if len(sys.argv) != 3:
        print 'Command e.g.: python cntBasicStatics.py -d 0(1)'
        sys.exit(1)

    para = parser.parse_args()
    if para.data_num == 0:
        eventinfo_path = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_1"]
        userevent_path = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_2"]
        userfriend_path = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_3"]
        usertag_path = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_4"]
        output_path = "./doubanSta.info"
    elif para.data_num == 1:
        eventinfo_path = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_1"]
        userevent_path = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_2"]
        output_path = "./doubanSta.info"
    if para.data_num == 2:
        eventinfo_path = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_1"]
        userevent_path = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_2"]
        output_path = "./doubanSta.info"
    else:
        print 'Invalid choice of dataset'
        sys.exit(1)

    uid_event_cnt = defaultdict(int)
    event_uid_cnt = defaultdict(int)
    for line in csv.reader(open(userevent_path)):
        uid = line[0]
        events = line[1].split(" ")
        for event in events:
            uid_event_cnt[uid] += 1
            event_uid_cnt[event] += 1

    print '1.User number:\t\t%d' % len(uid_event_cnt)
    print '2.Event number:\t\t%d' % len(event_uid_cnt)

    uid_event_dis = defaultdict(int)
    for uid in uid_event_cnt:
        uid_event_dis[uid_event_cnt[uid]]+=1
    uid_event_dis = sorted(uid_event_dis.items(), key=lambda x:x[1], reverse=True)
    wfd = open(output_path, "w")
    cum_prob = 0.0
    for pair in uid_event_dis:
        cum_prob += 1.0*pair[1]/len(uid_event_cnt)
        wfd.write("%s, %d, %.4f, %.4f\n" % (pair[0], pair[1], 1.0*pair[1]/len(uid_event_cnt), cum_prob))
    wfd.write('==================================================================\n')
    wfd.close()
    print '3.User attend event number distribution (see in "doubanSta.info")'

    event_uid_dis = defaultdict(int)
    for event in event_uid_cnt:
        event_uid_dis[event_uid_cnt[event]]+=1
    event_uid_dis = sorted(event_uid_dis.items(), key=lambda x:x[1], reverse=True)
    wfd = open(output_path, "a")
    cum_prob = 0.0
    for pair in event_uid_dis:
        cum_prob += 1.0*pair[1]/len(event_uid_cnt)
        wfd.write("%s, %d, %.4f, %.4f\n" % (pair[0], pair[1], 1.0*pair[1]/len(event_uid_cnt), cum_prob))
    wfd.close()
    print '4.Event has user number distribution (see in "doubanSta.info")'

    event_uid_dis = defaultdict(int)
    header = True
    total_num = 0
    try:
        for i, line in enumerate(csv.reader(open(eventinfo_path))):
            if header:
                header = False
                continue
            #eid = line[0]
            user_num = int(line[9])
            event_uid_dis[user_num] += 1
            total_num+=1
    except Exception, e:
        print e
        print i
        sys.exit(1)
    event_uid_dis = sorted(event_uid_dis.items(), key=lambda x:x[1], reverse=True)
    wfd = open(output_path, "a")
    wfd.write('==================================================================\n')
    cum_prob = 0.0
    for pair in event_uid_dis:
        cum_prob += 1.0*pair[1]/total_num
        wfd.write("%s, %d, %.4f, %.4f\n" % (pair[0], pair[1], 1.0*pair[1]/total_num, cum_prob))
    wfd.close()
    print '5.Event has real user number distribution (see in "doubanSta.info")'


if __name__ == "__main__":
    main()

