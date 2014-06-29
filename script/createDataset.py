#!/usr/bin/env python
#encoding=utf8

import csv, json, sys
from collections import defaultdict

reload(sys)
sys.setdefaultencoding('utf8')


VALID_CHINESE_SYMBOL = set(["。".decode("utf8"), "，".decode("utf8"),
    "《".decode("utf8"), "》".decode("utf8"), "“".decode("utf8"),
    "”".decode("utf8"), "、".decode("utf8"), "：".decode("utf8"),
    ".".decode("utf8"), "!".decode("utf8"), "！".decode("utf8")])


def is_symbol(uchar):
    if uchar in VALID_CHINESE_SYMBOL:
        return True
    return False


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False


def is_number(uchar):
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar<=u'\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
        return True
    else:
        return False


def is_other(uchar):
    """判断是否非汉字，数字和英文字符"""
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return True
    else:
        return False


def B2Q(uchar):
    """半角转全角"""
    inside_code=ord(uchar)
    if inside_code<0x0020 or inside_code>0x7e:      #不是半角字符就返回原来的字符
        return uchar
    if inside_code==0x0020: #除了空格其他的全角半角的公式为:半角=全角-0xfee0
        inside_code=0x3000
    else:
        inside_code+=0xfee0
    return unichr(inside_code)


def Q2B(uchar):
    """全角转半角"""
    inside_code=ord(uchar)
    if inside_code==0x3000:
        inside_code=0x0020
    else:
        inside_code-=0xfee0
    if inside_code<0x0020 or inside_code>0x7e:      #转完之后不是半角字符返回原来的字符
        return uchar
    return unichr(inside_code)


def stringQ2B(ustring):
    """把字符串全角转半角"""
    return "".join([Q2B(uchar) for uchar in ustring])


def uniform(ustring):
    """格式化字符串，完成全角转半角，大写转小写的工作"""
    return stringQ2B(ustring).lower()


def string2List(ustring):
    """将ustring按照中文，字母，数字分开"""
    retList=[]
    utmp=[]
    for uchar in ustring:
        if is_other(uchar):
            if len(utmp)==0:
                continue
            else:
                retList.append("".join(utmp))
                utmp=[]
        else:
            utmp.append(uchar)
    if len(utmp)!=0:
        retList.append("".join(utmp))
    return retList


def createEventData(infile, outfile1, outfile2):
    uid_event = defaultdict(list)
    writer = csv.writer(open(outfile1, "w"), lineterminator="\n")
    writer.writerow(["id", "city", "location", "coordinate", "title", "author",
        "category", "start_time", "end_time", "participants", "content"])
    recorded_event = set([])
    for line in open(infile):
        entry = line.strip("\r\t\n").split("\t")
        if len(entry) != 2:
            print 'Invalid segmentation.'
            sys.exit(1)
        uid = entry[0]
        clean_str = entry[1].replace("\\\\", "\\")
        parse_data = json.loads(clean_str)
        event_num = len(parse_data)
        for i in xrange(event_num):
            event_id = parse_data[i]["id"]["$t"].split("/")[-1]
            uid_event[uid].append(event_id)
            if event_id not in recorded_event:
                recorded_event.add(event_id)
                city = parse_data[i]["db:location"]["@id"]
                location = parse_data[i]["gd:where"]["@valueString"]
                coordinate = parse_data[i]["georss:point"]["$t"]
                title = parse_data[i]["title"]["$t"].replace(",", " ")
                author = parse_data[i]["author"]["name"]["$t"]
                category = parse_data[i]["category"]["@term"].split("#")[-1]
                start_time = parse_data[i]["gd:when"]["@startTime"]
                end_time = parse_data[i]["gd:when"]["@endTime"]
                participants = parse_data[i]["db:attribute"][2]["$t"]
                content = parse_data[i]["content"]["$t"]
                content = cleanContent(content)
                writer.writerow([event_id, city, location, coordinate, title,
                    author, category, start_time, end_time, participants, content])

    writer = csv.writer(open(outfile2, "w"), lineterminator="\n")
    writer.writerow(["uid", "event_id"])
    for uid in uid_event:
        events = " ".join(uid_event[uid])
        writer.writerow([uid, events])
    return uid_event


def createUserData(infile1, infile2, outfile1, outfile2):
    writer = csv.writer(open(outfile1, "w"), lineterminator="\n")
    for line in open(infile1):
        entry = line.strip("\r\t\n").split("\t")
        uid = entry[0]
        fuids = " ".join([fuid.strip("\\") for fuid in entry[1:]])
        writer.writerow([uid, fuids])
    writer = csv.writer(open(outfile2, "w"), lineterminator="\n")
    for line in open(infile2):
        uid_tag = []
        entry = line.strip("\r\t\n").split("\t")
        uid = entry[0]
        clean_str = entry[1].replace("\\\\", "\\")
        #clean_str = clean_str.replace("\"\"", "\"")
        parse_data = json.loads(clean_str)
        for i in xrange(len(parse_data)):
            uid_tag.append(parse_data[i]["title"]["$t"])
        writer.writerow([uid]+uid_tag)


def cleanContent(content):
    clean_str=""
    content = content.replace("\r\n", "")
    content = content.replace("\n", "")
    content = content.replace("  ", " ")
    content = content.replace(",", "，")
    content = content.decode("utf8")
    for i in xrange(len(content)):
        if is_symbol(content[i]) or is_number(content[i])\
            or is_chinese(content[i]) or is_alphabet(content[i]):
                clean_str += content[i].encode("utf8")
        else:
            clean_str += " "
    clean_str = "，".join(clean_str.strip(" ").split())
    return clean_str


if __name__ == "__main__":
    infile = "Event/doubanEvent.csv"
    outfile1 = "Event/eventInfo.csv"
    outfile2 = "Event/userEvent.csv"
    user_event = createEventData(infile, outfile1, outfile2)

    #infile1 = "Event/doubanFollows.csv"
    #infile2 = "Event/doubanUserTags.csv"
    #outfile1 = "Event/userFriend.csv"
    #outfile2 = "Event/userTag.csv"
    #createUserData(infile1, infile2, outfile1, outfile2)
