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
# Extract data for "Beijing" and "Shanghai"                       #
###################################################################

import csv, json, sys
from collections import defaultdict

with open("../SETTINGS.json") as fp:
    settings = json.loads(fp.read())


def extract_data(event_file, city_event_file, city_name):
    wfd = open(city_event_file, "w")
    for i, line in enumerate(open(event_file)):
        name = line.strip("\r\t\n").split(",")[1]
        if city_name == name:
            wfd.write(line)
    wfd.close()

def main():
    eventinfo_path = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_1"]
    city_eventinfo_path = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_CITY1"]
    city_name = "beijing"
    extract_data(eventinfo_path, city_eventinfo_path, city_name)

    city_eventinfo_path = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_CITY2"]
    city_name = "shanghai"
    extract_data(eventinfo_path, city_eventinfo_path, city_name)

if __name__ == "__main__":
    main()
