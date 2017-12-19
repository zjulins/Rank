'''
Created on Mar 15, 2016

@author: weilin2
'''
max_men_wn = 10;
max_doc_wn = 800;
max_att_wn = 100;

class Sample(object):
    def __init__(self, infoStr = ''):
        self.rank_value  = 0
        self.mentionID   = ''
        self.mentionType = ''
        self.headStr     = ''
        self.docID       = ''
        self.kbNodeID    = ''
        self.kbNodeType  = ''
        self.kbNodeStr   = ''
        self.wikiPageID  = ''
        #self.orgInfoStr  = ''
        self.predictScore = 0.0
        
        self.loadFromStr(infoStr)
    
    
    def loadFromStr(self, infoStr):
        if(infoStr != ''):
            #self.orgInfoStr = infoStr
            terms = infoStr.strip().split('\t')  
            if(len(terms) != 10):
                print 'Sample.loadFromStr err!'
                print len(terms), infoStr
                for t in terms:
                    print t
                exit
            self.rank_value  = int(float(terms[0]));
            self.mentionID   = terms[1]
            self.mentionType = terms[2]
            self.headStr     = terms[3][1:-1]
            self.docID       = terms[4]
            self.kbNodeID    = terms[5]
            self.kbNodeType  = terms[6][1:-1]
            self.kbNodeStr   = terms[7][1:-1]
            self.wikiPageID  = terms[8][1:-1]            
    
    def toString(self):
        return str(self.rank_value) + "\t" + self.mentionID + "\t" + self.mentionType \
            + "\t[" + self.headStr + ']\t' + self.docID + "\t" + self.kbNodeID + "\t[" + self.kbNodeType \
            + "]\t[" + self.kbNodeStr + "]\t[" + self.wikiPageID + "]"    
    def toValueString(self):
        return 'rank_value_v:' + str(self.rank_value_v) + "\n" \
            + 'mentionType_v:' + str(self.mentionType_v) + "\n" \
            + 'headStr_v:' + str(self.headStr_v) + '\n' \
            + 'docContext_v' + str(self.docContext_v) + '\n' \
            + 'kbNodeID_v' + str(self.kbNodeID_v) + '\n' \
            + 'kbNodeType_v' + str(self.kbNodeType_v) + '\n' \
            + 'kbNodeStr_v' + str(self.kbNodeStr_v) + '\n' \
            + 'wikiContext_v' + str(self.wikiContext_v)  
    def getValues(self):
        x = []
        x.extend(self.headStr_v)
        x.extend(self.mentionType_v)
        x.extend(self.docContext_v)
        x.extend(self.kbNodeID_v)
        x.extend(self.kbNodeStr_v)
        x.extend(self.kbNodeType_v)
        x.extend(self.wikiContext_v)
        x.extend(self.rank_value_v)
        x.extend(self.stringMatch_v)
        return x
    def getValueDim(self):
        return max_men_wn * 2 + 1 + max_doc_wn * 2 + 1 + max_men_wn * 2 + max_men_wn * 2 +  max_doc_wn * 2 + 1 + 1
    