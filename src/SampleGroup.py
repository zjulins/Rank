'''
Created on Mar 22, 2016

@author: weilin2
'''

from Sample import *

class SampleGroup(object):
    '''
    classdocs
    '''
    def __init__(self):
        self.samples        = []
        self.groupName      = '' 
        self.groupID        = -1
        self.centerPosition = 0
        self.positiveSample=[]
        self.negativeSample = []
        NIL_sample = Sample()
        NIL_sample.headStr  = 'NIL'
        NIL_sample.kbNodeID = 'NIL'
        self.samples.append(NIL_sample)

    def getBoostIdx(self):
        boostIdx = 0;
        for i in range(len(self.samples)):
            if(self.samples[i].rank_value == 1):
                boostIdx = i;
                break;
        return boostIdx;

    def getSampleCount(self):
        return len(self.samples)


    def printInfoStr(self):
        print 'group name: ', self.groupName
        for s in self.samples:
            print s.toString()

    def setCenterPosition(self,mentionID):
        pos1=mentionID.find(':')
        pos2=mentionID.find('-')
        beginPosition=int(mentionID[pos1+1:pos2])
        endPosition=int(mentionID[pos2+1:])
        self.centerPosition=(beginPosition+endPosition)/2

    def getKbNodes(self):
        nodes=[]
        for sample in self.samples:
            nodes.append(sample.kbNodeID)
        return nodes
