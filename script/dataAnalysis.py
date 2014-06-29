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
# Providing various functions for data analysis. These results    #
#   support our final model choice.                               #
###################################################################

import sys, csv, json, argparse, datetime
from collections import defaultdict

with open("../SETTINGS.json") as fp:
    settings = json.loads(fp.read())


def cnt_num_attendant(eventinfo_path, staresult_path):
    ''' count the distribution of number of attendants '''
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


def cnt_attendant_for_category(eventinfo_path, staresult_path):
    ''' count number of categories and the distribution of number of
        attendants for each category
    '''
    category_numevents = defaultdict(int)
    category_numattendants = defaultdict(int)
    for i, line in enumerate(open(eventinfo_path)):
        category = line.strip("\r\t\n").split(",")[6]
        num_participants = int(line.strip("\r\t\n").split(",")[9])
        category_numevents[category] += 1
        category_numattendants[category] += num_participants
    print "Category statistics information--------\n"
    print "\tNumber of categories: %d" % len(category_numevents)
    wfd = open(staresult_path, "w")
    for category in category_numevents:
        wfd.write("%s %d %f\n" % (category, 1.0*category_numattendants[category]/category_numevents[category]))
    print 'Average number of attendants for each category can be seen in (staresult.txt)'


def cnt_attendant_for_location(eventinfo_path, staresult_path):
    ''' count number of locations and the average of number of
        attendants for each location
    '''
    stopwords = set([])
    for line in open(settings["ROOT_PATH"]+settings["STOP_WORD_FILE"]):
        term = line.strip("\r\t\n")
        stopwords.add(term)

    term_numevents = defaultdict(int)
    term_numattendants = defaultdict(int)
    for i, line in enumerate(open(eventinfo_path)):
        location = line.strip("\r\t\n").split(",")[2]
        num_participants = int(line.strip("\r\t\n").split(",")[9])
        for term in set(location.split(" ")):
            term = term.decode("utf8")
            tag = True
            for j in xrange(len(term)):
                if not is_chinese(term[j]):
                    tag = False
            if not tag:
                continue
            term = term.encode("utf8")
            if term in stopwords:
                continue
            term_numevents[term] += 1
            term_numattendants[term] += num_participants
    term_numevents1 = sorted(term_numevents.items(), key=lambda x:x[1], reverse=True)
    term_aveattendants = defaultdict(int)
    for pair in term_numevents1[:settings["MAX_LOCATIONS"]]:
        term_aveattendants[pair[0]] = 1.0*term_numattendants[pair[0]]/pair[1]

    term_aveattendants = sorted(term_aveattendants.items(), key=lambda x:x[1], reverse=True)
    wfd = open(staresult_path, "w")
    for pair in term_aveattendants:
        wfd.write("%s %d %f\n" % (pair[0], term_numevents[pair[0]], pair[1]))
    wfd.close()


def cnt_attendant_for_organizer(eventinfo_path, staresult_path, min_support=1):
    ''' count number of organizors and the average of number of
        attendants for each organizor
    '''
    organizor_numevents = defaultdict(int)
    organizor_numattendants = defaultdict(int)
    for i, line in enumerate(open(eventinfo_path)):
        organizor = line.strip("\r\t\n").split(",")[5]
        num_participants = int(line.strip("\r\t\n").split(",")[9])
        organizor_numevents[organizor] += 1
        organizor_numattendants[organizor] += num_participants
    print "Organizor statistics information--------\n"
    print "\tNumber of categories: %d" % len(organizor_numevents)
    wfd = open(staresult_path, "w")
    for organizor in organizor_numevents:
        if (organizor_numevents[organizor]) >= min_support:
            wfd.write("%s %d %f\n" % (organizor, organizor_numevents[organizor],
                1.0*organizor_numattendants[organizor]/organizor_numevents[organizor]))
    print 'Average number of attendants for each organizor can be seen in (staresult.txt)'


def cnt_attendant_for_time(eventinfo_path, staresult_path):
    ''' count the number of attendants for each time period:
        (morning, afternoon, evening, other) * (weekday, weekend) + multiple days
        + multiple weeks + multiple month.
        More specifically, (morning, weekday):0, (afternoon, weekday):1, ...
    '''
    timeperiod_numevents = defaultdict(int)
    timeperiod_numattendants = defaultdict(int)
    for i, line in enumerate(open(eventinfo_path)):
        start_time = line.strip("\r\t\n").split(",")[7]
        end_time = line.strip("\r\t\n").split(",")[8]
        num_participants = int(line.strip("\r\t\n").split(",")[9])
        timeidx = getIdOfTimePeriod(start_time, end_time)
        timeperiod_numevents[timeidx] += 1
        timeperiod_numattendants[timeidx] += num_participants
    print "Time statistics information--------\n"
    print "\tNumber of categories: %d" % len(timeperiod_numevents)
    wfd = open(staresult_path, "w")
    for timeperiod in timeperiod_numevents:
        wfd.write("%s %d %f\n" % (timeperiod, timeperiod_numevents[timeperiod],
            1.0*timeperiod_numattendants[timeperiod]/timeperiod_numevents[timeperiod]))
    print 'Average number of attendants for each timeperiod can be seen in (staresult.txt)'


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


def cnt_attendant_for_intro(eventinfo_path, staresult_path):
    ''' count the number of events the specified term occur in and
        the average number of attendants for each term
    '''
    stopwords = set([])
    for line in open(settings["ROOT_PATH"]+settings["STOP_WORD_FILE"]):
        term = line.strip("\r\t\n")
        stopwords.add(term)

    term_numevents = defaultdict(int)
    term_numattendants = defaultdict(int)
    for i, line in enumerate(open(eventinfo_path)):
        intro = line.strip("\r\t\n").split(",")[10]
        num_participants = int(line.strip("\r\t\n").split(",")[9])
        for term in set(intro.split(" ")):
            term = term.decode("utf8")
            tag = True
            for j in xrange(len(term)):
                if not is_chinese(term[j]):
                    tag = False
            if not tag:
                continue
            term = term.encode("utf8")
            if term in stopwords:
                continue
            term_numevents[term] += 1
            term_numattendants[term] += num_participants
    term_numevents1 = sorted(term_numevents.items(), key=lambda x:x[1], reverse=True)
    term_aveattendants = defaultdict(int)
    for pair in term_numevents1[:settings["MAX_WORDS"]]:
        term_aveattendants[pair[0]] = 1.0*term_numattendants[pair[0]]/pair[1]

    term_aveattendants = sorted(term_aveattendants.items(), key=lambda x:x[1], reverse=True)
    wfd = open(staresult_path, "w")
    for pair in term_aveattendants:
        wfd.write("%s %d %f\n" % (pair[0], term_numevents[pair[0]], pair[1]))
    wfd.close()


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False

def cnt_attendant_for_title(eventinfo_path, staresult_path):
    ''' count the number of events the specified term occur in and
        the average number of attendants for each term
    '''
    stopwords = set([])
    for line in open(settings["ROOT_PATH"]+settings["STOP_WORD_FILE"]):
        term = line.strip("\r\t\n")
        stopwords.add(term)

    term_numevents = defaultdict(int)
    term_numattendants = defaultdict(int)
    for i, line in enumerate(open(eventinfo_path)):
        title = line.strip("\r\t\n").split(",")[4]
        num_participants = int(line.strip("\r\t\n").split(",")[9])
        for term in set(title.split(" ")):
            term = term.decode("utf8")
            tag = True
            for j in xrange(len(term)):
                if not is_chinese(term[j]):
                    tag = False
            if not tag:
                continue
            term = term.encode("utf8")
            if term in stopwords:
                continue
            term_numevents[term] += 1
            term_numattendants[term] += num_participants
    term_numevents1 = sorted(term_numevents.items(), key=lambda x:x[1], reverse=True)
    term_aveattendants = defaultdict(int)
    for pair in term_numevents1[:settings["MAX_WORDS"]]:
        term_aveattendants[pair[0]] = 1.0*term_numattendants[pair[0]]/pair[1]

    term_aveattendants = sorted(term_aveattendants.items(), key=lambda x:x[1], reverse=True)
    wfd = open(staresult_path, "w")
    for pair in term_aveattendants:
        wfd.write("%s %d %f\n" % (pair[0], term_numevents[pair[0]], pair[1]))
    wfd.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, action='store',
            dest='data_num', help='choose which data set to use')
    parser.add_argument('-f', type=int, action='store',
            dest='function_num', help='choose which data analysis function to use')
    if len(sys.argv) != 5:
        print 'Command e.g.: python segmentData.py -d 1(11,12,...) -f 1(2,3,...)'
        sys.exit(1)

    para = parser.parse_args()
    if para.data_num == "1":
        eventinfo_path = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_1"]
        staresult_path = "./staresult.txt"
    elif para.data_num == "11":
        eventinfo_path = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_CITY1"]
        staresult_path = "./staresult.txt"
    elif para.data_num == "12":
        eventinfo_path = settings["ROOT_PATH"] + settings["SRC_DATA_FILE1_CITY2"]
        staresult_path = "./staresult.txt"
    else:
        print 'Invalid choice of dataset'
        sys.exit(1)

    if para.function_num == 1:
        cnt_num_attendant(eventinfo_path, staresult_path)
    elif para.function_num == 2:
        cnt_attendant_for_category(eventinfo_path, staresult_path)
    elif para.function_num == 3:
        cnt_attendant_for_location(eventinfo_path, staresult_path)
    elif para.function_num == 4:
        cnt_attendant_for_organizer(eventinfo_path, staresult_path, 5)
    elif para.function_num == 5:
        cnt_attendant_for_time(eventinfo_path, staresult_path)
    elif para.function_num == 6:
        cnt_attendant_for_intro(eventinfo_path, staresult_path)
    elif para.function_num == 7:
        cnt_attendant_for_title(eventinfo_path, staresult_path)


if __name__ == "__main__":
    main()

