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
# Combine event raw attributes info with intro text segmentation  #
#   results.                                                      #
###################################################################

import csv
from collections import defaultdict

def main():
    infile1 = "./Event/eventInfo.csv"
    infile2 = "./Event/eventInfo.WDS.csv"
    outfile1 = "/home/anthonylife/Doctor/Code/MyPaperCode/EventPopularity/data/eventInfo.csv"

    event_info = defaultdict(list)
    idx = 0
    for line in open(infile1):
        parts = line.strip("\r\t\n").split(",")
        if idx == 0:
            header = parts
            idx += 1
            continue
        eid = parts[0]
        event_info[eid] = parts[1:10]

    header= header + ["entity"]
    writer = csv.writer(open(outfile1, "w"), lineterminator="\n")
    writer.writerow(header)
    for line in open(infile2):
        parts = line.strip("\r\t\n").split(",")
        eid = parts[0]
        info_content = parts[2]
        entity_content = parts[4]
        writer.writerow([eid]+event_info[eid]+[info_content]+[entity_content])

if __name__ == "__main__":
    main()

