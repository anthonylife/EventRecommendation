#!/usr/bin/env python
#encoding=utf8

if __name__ == "__main__":
    infile="/home/anthonylife/Doctor/Data/Douban/Event/eventInfo.csv"
    outfile = "/home/anthonylife/Doctor/Code/MyPaperCode/EventRecommendation/data/doubanEventInfo.csv"
    wfd = open(outfile, "w")
    for line in open(infile):
        line = line.replace("^M", "")
        wfd.write(line)
    wfd.close()
