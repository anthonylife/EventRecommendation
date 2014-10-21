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
# Utilize Baidu Map API to correct some invalid geo info of POI.  #
###################################################################


import urllib2, json, sys, csv, time

SETTINGS = json.loads(open("../SETTINGS.json").read())

# Interface parameter of Baidu Geocoding API v2.0
API_KEY = 'N4y6bK1vzMEi4Qf0ZNecS6TB'
URL_GET_BASE = 'http://api.map.baidu.com/geocoder/v2/?'
OUTPUT = 'json'
MAX_REQ_NUM = 10


def checkGeo(geoaddr, valid_geo):
    if geoaddr[0] > valid_geo[0] and geoaddr[0] < valid_geo[1]\
            and geoaddr[1] > valid_geo[2] and geoaddr[1] < valid_geo[3]:
        return True
    return False


def callBaiduMapAPI(geoname, cityname):
    tried_num = 0
    while True:
        result = urllib2.urlopen("%saddress=%s&city=%s&output=%s&ak=%s" % (URL_GET_BASE, geoname, cityname, OUTPUT, API_KEY))
        content = result.read().strip()
        tried_num = 0
        if len(content) == 0:
            time.sleep(tried_num)
            continue
        print content
        print len(content)
        break


def correctGeoInfo(valid_geo, input_file, output_file, cityname):
    total_num = 0
    valid_num = 0
    correct_num = 0
    for entry in csv.reader(open(input_file, "r"), lineterminator="\n"):
        geoname = entry[2].replace(" ", "")
        geoaddr = map(float, entry[3].split(" "))
        total_num += 1
        if checkGeo(geoaddr, valid_geo):
            valid_num += 1
            continue
        else:
            geoaddr = callBaiduMapAPI(geoname, cityname)
    print total_num
    print valid_num


if __name__ == "__main__":
    callBaiduMapAPI('北京工人体育场', '北京')
    callBaiduMapAPI('北京保利影院、南京', '北京')

    '''# Beijing
    valid_geo =[39.5, 41, 115.5, 117]
    input_file = SETTINGS["ROOT_PATH"]+SETTINGS["SRC_DATA_FILE1_CITY1"]
    output_file = "./tmp1.csv"
    correctGeoInfo(valid_geo, input_file, output_file, '北京')

    # Shanghai
    valid_geo =[30.5, 31.7, 120.8, 122.1]
    input_file = SETTINGS["ROOT_PATH"]+SETTINGS["SRC_DATA_FILE1_CITY2"]
    output_file = "./tmp2.csv"
    correctGeoInfo(valid_geo, input_file, output_file, '上海')'''

