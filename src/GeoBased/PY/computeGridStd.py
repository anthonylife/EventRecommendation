
import sys, json, random
sys.path.append("../../lib/geopy-1.3.0/")
import numpy as np
from geopy.distance import GreatCircleDistance

settings = json.loads(open("../../SETTINGS.json").read())


if __name__ == "__main__":
    geo_start = [39.80, 116.17]
    geo_range = [0.013, 0.013]

    pois_num = 100
    pois = np.array([[0.0, 0.0] for i in xrange(pois_num)])
    for i in xrange(pois_num):
        pois[i] = np.array([geo_start[0]+geo_range[0]*random.random(), geo_start[1]+geo_range[1]*random.random()])

    distance_samples = []
    for i in xrange(pois_num):
        for j in xrange(i+1, pois_num):
            distance = GreatCircleDistance(pois[i], pois[j]).kilometers
            distance_samples.append(distance)
    distance_samples = np.array(distance_samples)
    print np.std(distance_samples)
