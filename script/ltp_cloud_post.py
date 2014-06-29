#!/usr/bin/enr python
# -*- coding: utf-8 -*-

# This example shows how to use Python to access the LTP API to perform full
# stack Chinese text analysis including word segmentation, POS tagging, dep-
# endency parsing, name entity recognization and semantic role labeling and
# get the result in specified format.
# We user the LTP API to segment the event text and extract entities from it.

import urllib2, urllib
import sys, csv, json

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ["xml", "json", "conll"]:
        print >> sys.stderr, "usage: %s [xml/json/conll]" % sys.argv[0]
        sys.exit(1)

    infile = "eventInfo.csv"
    outfile = "eventIntroAnalysis.csv"
    uri_base = "http://api.ltp-cloud.com/analysis/?"
    api_key = "H1k4M2H7FQplWlOKgw7yEaEB2Rhzhh4xHZ9H9sYe"

    wfd = csv.writer(open(outfile, "w"), lineterminator="\n")
    for i, entry in enumerate(csv.reader(open(infile))):
        if i == 0:
	    continue
	event_id = entry[0]
        whole_text = entry[10]

	# Note that if your text contain special characters such as linefeed or '&',
        # you need to use urlencode to encode your data
	whole_text = whole_text.decode("utf8")
	tokens = []
        entities = set([])
	for j in xrange(0, len(whole_text)-100, 100):
            text = whole_text[j:j+100]	
	    text = text.encode("utf8")
	    text = urllib.quote(text)
            format = sys.argv[1]
            pattern = "ner"
            values = {"api_key": api_key, "text": text, "format": format,
                "pattern": pattern}
            data = urllib.urlencode(values)
        
            ne_conseq_tag = False
            long_entity = ""
            try:
                req = urllib2.Request(uri_base, data)
                try:
	    	    response = urllib2.urlopen(req)
                except urllib2.HTTPError, e:
		    print >> i, sys.stderr, e.reason
		    continue 
	        a = response.read()
                content = json.loads(a.strip())
                para_num =len(content)
                for para_id in xrange(para_num):
                    senten_num = len(content[para_id])
                    for senten_id in xrange(senten_num):
                        token_num = len(content[para_id][senten_id])
                        for token_id in xrange(token_num):
                            pos = content[para_id][senten_id][token_id]["pos"].encode("utf-8")
                            token = content[para_id][senten_id][token_id]["cont"].encode("utf-8")
                            if pos == "wp":
                                continue
                            tokens.append(token)
                            if pos == "ns":
                                entities.add(token)
                                if ne_conseq_tag == True:
                                    long_entity += token
                                else:
				    long_entity = token
                                    tag = True
                            else:
                                if ne_conseq_tag == True:
                                    entities.add(long_entity)
                                    ne_conseq_tag = False
            except urllib2.HTTPError, e:
	        print >> i, sys.stderr, e.reason, e
		continue
        wfd.writerow([event_id] +[len(tokens)] + [" ".join(list(tokens))] + [len(entities)] + [" ".join(list(entities))])
        sys.stdout.write("\rFINISHED PAIR NUM: %d... " % (i+1))
        sys.stdout.flush()

