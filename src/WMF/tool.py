import random, time

def rZero(k):
    return [0.0 for i in range(k)]


def rGaussian(k):
    factor = [random.normalvariate(0, 0.01) for i in xrange(k)]
    for i in xrange(len(factor)):
        if factor[i] > 1:
            factor[i] = 1
        elif factor[i] < -1:
            factor[i] = -1
    return factor


def rPosGaussian(k):
    factor = [random.normalvariate(0, 0.01) for i in xrange(k)]
    for i in xrange(len(factor)):
        if factor[i] > 1:
            factor[i] = 1
        elif factor[i] < -1:
            factor[i] = -1
        factor[i] = (factor[i]+1)/2.0
    return factor

def tic():
    globals()['tt'] = time.clock()

def toc():
    return time.clock()-globals()['tt']

