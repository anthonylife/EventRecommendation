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
# Date: 2014/6/14                                                 #
# Count distribution of number of users attending activities.     #
###################################################################

import csv, json, sys
from collections import defaultdict

with open("../SETTINGS.json") as fp:
    settings = json.loads(fp.read())

def main():
    eventinfo_path = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_1"]
    staresult_path = "./staresult.txt"

    num_attendant = defaultdict(int)
    total_num = 0
    for i, line in enumerate(open(eventinfo_path)):
        try:
            num = int(line.strip("\r\t\n").split(",")[9])
            num_attendant[num] += 1
            total_num += 1
        except:
            print line
            print i
            sys.exit(1)

    cum_prob = 0.0
    num_attendant = sorted(num_attendant.items(), key=lambda x:x[0])
    wfd = open(staresult_path, "w")
    for pair in num_attendant:
        cum_prob += 1.0*pair[1]/total_num
        wfd.write("%d %d %.4f\n" % (pair[0], pair[1], cum_prob))
    wfd.close()

if __name__ == "__main__":
    main()

