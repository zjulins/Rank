#coding=utf-8
'''
Created on Mar 17, 2016

@author: weilin2
'''


from Sample import *
import random
import numpy
import SampleGroup
import StringMatcher
import DocMatcher
import copy
from Vocab import *

class DataProvider(object):

    def __init__(self, lang,  filePath, hotFilePath, muti_name_file, att_file, batch_size, cache_size, maxSampCount, shuffle, word_vocab, kb_vocab, kbp_type_vocab,kb_type_vocab):
        self.lang        = lang
        assert(self.lang == 'ENG' or self.lang == 'CMN' or self.lang == 'SPA')
        self.doc_avg_dis = {}
        self.doc_max_dis = {}
		#para1
        self.doc_avg_dis['ENG'] = 0.0445
        self.doc_avg_dis['CMN'] = 0
        self.doc_avg_dis['SPA'] = 0.1397
        self.doc_max_dis['ENG'] = 0.2
        self.doc_max_dis['CMN'] = 0.8
        self.doc_max_dis['SPA'] = 0.3

		#para2
        #self.doc_avg_dis['ENG'] = 0.0
        #self.doc_avg_dis['CMN'] = 0.0
        #self.doc_avg_dis['SPA'] = 0.1397
        #self.doc_max_dis['ENG'] = 1.0
        #self.doc_max_dis['CMN'] = 1.0
        #self.doc_max_dis['SPA'] = 0.3

        self.filePath    = filePath
        self.hotFilePath = hotFilePath
        self.mutiNamePath= muti_name_file
        self.att_file    = att_file
        self.shuffle     = shuffle
        self.stopWord    = {}
        self.wikiContext = {}
        self.docContext  = {}
        self.wikiContextIDs = {}
        self.docContextIDs  = {}
        self.wordsINDoc     = {}
        self.wordsINWiki    = {}
        self.hotValue       = {}
        self.samples        = []
        self.candAttWord    = {} 
        self.mutiName       = {}


        self.word_vocab     = word_vocab
        self.kb_vocab       = kb_vocab
        self.kbp_type_vocab = kbp_type_vocab
        self.kb_type_vocab  = kb_type_vocab
        
        self.batch_size  = batch_size
        self.cache_size  = cache_size
        self.FileEnd  = 0
        self.file        = file
        self.samples_count = 0
        self.stringMatcher = StringMatcher.StringMatcher()
        self.docMatcher    = DocMatcher.DocMatcher()

        self.maxSampCount    = maxSampCount
        self.group           = None

        if(self.lang == 'ENG'):
            self.docMatcher._loadIDF('../file/idf.txt', self.word_vocab)
            self.__loadStopWord('../file/StopWord.txt')
            self.__readDocWikiContext(self.filePath)
            self.__loadHotFile(self.hotFilePath)
            self.__loadMutiName(self.mutiNamePath)
        elif(self.lang == 'CMN'):
            self.docMatcher._loadIDF('../data/IDF.txt', self.word_vocab)
            self.__readDocWikiContext(self.filePath)
            self.__loadHotFile(self.hotFilePath)
            self.__loadMutiName(self.mutiNamePath)
        elif(self.lang == 'SPA'):
            self.docMatcher._loadIDF('../file/idf.txt', self.word_vocab)
            self.__readDocWikiContext(self.filePath)
            self.__loadHotFile(self.hotFilePath)
            self.__loadMutiName(self.mutiNamePath)

        self.reset()
    
    def __loadMutiName(self, filePath):
        print 'load mutiName from ', filePath
        file = open(filePath, 'rb')
        loaded = 0
        for line in file.readlines():
            line  = line.decode('utf8').strip()
            terms = line.split('\t');
            if(len(terms) != 2):
                print '__loadMutiName warn:', line.encode('utf8')
                continue
            nid       = terms[0]
            name      = terms[1]
            if(not self.mutiName.has_key(nid)):
                self.mutiName[nid] = {}
            self.mutiName[nid][name] = 1
            loaded += 1
        file.close()
        print 'load mutiName count: ', loaded

    def __loadAttWord(self, filePath):
        print 'load att word from ', filePath
        file = open(filePath, 'rb')
        loaded = 0
        for line in file.readlines():
            line  = line.decode('utf8').strip()
            terms = line.split('\t');
            men_id    = terms[0]
            cand      = terms[1]
            id        = men_id + cand
            if(not self.candAttWord.has_key(id)):
                self.candAttWord[id] = []
            menOrgStr = terms[2].lower()
            mens      = menOrgStr.split(', ')
            for m in mens:
                words = m.split(' ')
                for w in words:
                    self.candAttWord[id].append(self.word_vocab.search(w))
            loaded += 1
        file.close()
        print 'load att word count: ', loaded
    
    def isEnd(self):
        if((self.FileEnd == 1) and (not self.samples)):
            return 1
        else:
            return 0
    
    def reset(self):
        self.file     = open(self.filePath, 'rb')
        self.FileEnd  = 0
        self.group    = None
    
    def __loadStopWord(self, filePath):
        print 'load stop word from ', filePath
        file = open(filePath, 'rb')
        loaded = 0
        for line in file.readlines():
            line  = line.decode('utf8').strip()
            self.stopWord[line.lower()] = 1
            loaded += 1
        file.close()
        print 'load stop word count: ', loaded
    
    def __loadHotFile(self, filePath):
        print 'DataProvider.loadHotFile begin'
        file = open(filePath, 'rb')
        total_count = 0
        loaded      = 0
        for line in file.readlines():
            line  = line.decode('utf8').strip()
            info  = line.split()
            if(len(info) == 2 and self.hotValue.has_key(info[0])):
                self.hotValue[info[0]] = float(info[1])
                loaded += 1
            total_count += 1
        file.close()
        print 'DataProvider.loadHotFile end. total %d lines, load %d lines' %(total_count, loaded)
        
    
    def __readDocWikiContext(self, file_path, wikiExtraFilePath = None):
        file = open(file_path, 'rb')
        doc_begin  = 0
        wiki_begin = 0
        while(1):
            line = file.readline().decode('utf8').strip()
            if(not line):
                break;
            if(line == u'=== doc ==='):
                doc_begin = 1
                continue
            if(line == u'=== wiki ==='):
                doc_begin = 0
                wiki_begin = 1
                continue
            if(doc_begin == 0 and wiki_begin == 0):
                if(line.startswith(u'MentionID:')):
                    self.samples_count += 1
                else:
                    terms = line.strip().split('\t')  
                    self.hotValue[terms[5]] = 0
                
            
            if(doc_begin == 1):
                id      = line[4:]

                line    = file.readline().decode('utf8')
                context = line[9:]
                self.wordsINDoc[id]  = {}
                tmp = []

                if(self.lang == 'ENG' or self.lang == 'SPA'):
                    for item in context.strip().split():
                        if(self.stopWord.has_key(item.lower())):
                            continue
                        wid = self.word_vocab.search(item.lower());
                        if(wid == self.word_vocab.unkID):
                            continue
                        else:
                            tmp.append(wid)
                            self.wordsINDoc[id][wid] = 1
                elif(self.lang == 'CMN'):
                    for i in range(len(context)):
                        if(self.stopWord.has_key(context[i])):
                            continue
                        wid = self.word_vocab.search(context[i]);
                        if(wid == self.word_vocab.unkID):
                            continue
                        else:
                            tmp.append(wid)
                            self.wordsINDoc[id][wid] = 1

                self.docContextIDs[id] = tmp
                
            if(wiki_begin == 1):
                id      = line[4:]

                line    = file.readline().decode('utf8').strip()
                context = line[9:]

                self.wordsINWiki[id]  = {}
#                 self.wikiContext[id] = context
                tmp = []
                if(self.lang == 'ENG' or self.lang == 'SPA'):
                    for item in context.strip().split():
                        if(self.stopWord.has_key(item.lower())):
                            continue
                        wid = self.word_vocab.search(item.lower());
                        if(wid == self.word_vocab.unkID):
                            continue
                        else:
                            tmp.append(wid)
                            self.wordsINWiki[id][wid] = 1
                elif(self.lang == 'CMN'):
                    for i in range(len(context)):
                        if(self.stopWord.has_key(context[i])):
                            continue
                        wid = self.word_vocab.search(context[i]);
                        if(wid == self.word_vocab.unkID):
                            continue
                        else:
                            tmp.append(wid)
                            self.wordsINWiki[id][wid] = 1


                self.wikiContextIDs[id] = tmp
        self.wordsINWiki[u'<null>']  = {}
        self.wikiContextIDs[u'<null>'] = []
        print 'total %d samples' %(self.samples_count)
        file.close()

    
    def fillValue(self, aBatch):
        self.maxSampCount = 0
        for i in range(self.batch_size):
            if(self.maxSampCount < len(aBatch[i].samples)):
                self.maxSampCount = len(aBatch[i].samples)
        if(self.maxSampCount > 100 and self.shuffle == 1):
            self.maxSampCount = 100

        fea  = numpy.zeros((self.batch_size, self.maxSampCount - 1, 11 + max_men_wn * 6), dtype = 'int32')
        lab  = numpy.zeros((self.batch_size, self.maxSampCount, 2), dtype = 'int32')
        
        for i in range(self.batch_size):
            aGroup = self.shuffleGroup(aBatch[i])
            node_hot = numpy.zeros((self.maxSampCount), dtype = 'float32')
            for j in range(len(aGroup.samples) - 1):
                #mention type
                fea[i,j, 0] = self.kbp_type_vocab.search(aGroup.samples[j+1].mentionType)    #kbp_type_vocab
                #kbNodeType
                k = 0
                for item in aGroup.samples[j+1].kbNodeType.strip().split(', '):
                    dot_index = item.find('.')
                    if(dot_index >= 0):
                        item = item[:dot_index]
                    fea[i,j,2 + k] = self.kb_type_vocab.search(item) 
                    fea[i,j,2 + max_men_wn + k] = 1
                    k = k + 1
                    if(k >= max_men_wn):
                        break 
                #headStr
                k = 0
                menWordsINWikiValue = 0
                if(self.lang == 'ENG' or self.lang == 'SPA'):
                    for item in aGroup.samples[j+1].headStr.strip().split():
                        w                                = self.word_vocab.search(item.lower())
                        fea[i,j,2 + 2 * max_men_wn + k]  = w
                        fea[i,j,2 + 3 * max_men_wn + k]  = 1
                        if(self.wordsINWiki[aGroup.samples[j+1].wikiPageID].has_key(w)):
                            menWordsINWikiValue += 1
                        k = k + 1
                        if(k >= max_men_wn):
                            break
                    if(k == 0):
                        menWordsINWikiValue              = 0
                    else:
                        menWordsINWikiValue = menWordsINWikiValue * 1.0 / k
                elif(self.lang == 'CMN'):
                    for q in range(len(aGroup.samples[j+1].headStr.strip())):
                        w                                = self.word_vocab.search(aGroup.samples[j+1].headStr[q])
                        fea[i,j,2 + 2 * max_men_wn + k]  = w
                        fea[i,j,2 + 3 * max_men_wn + k]  = 1
                        if(self.wordsINWiki[aGroup.samples[j+1].wikiPageID].has_key(w)):
                            menWordsINWikiValue += 1
                        k = k + 1
                        if(k >= max_men_wn):
                            break
                    if(k == 0):
                        menWordsINWikiValue              = 0                #统计mention的每个词在candidata中的比例
                    else:
                        menWordsINWikiValue = menWordsINWikiValue * 1.0 / k

                #kbNodeStr
                k = 0
                candWordsINDocValue = 0
                if(self.lang == 'ENG' or self.lang == 'SPA'):
                    for item in aGroup.samples[j+1].kbNodeStr.strip().split():
                        w                               = self.word_vocab.search(item.lower()) 
                        fea[i,j,2 + 4 * max_men_wn + k] = w
                        fea[i,j,2 + 5 * max_men_wn + k] = 1
                        if(self.wordsINDoc[aGroup.samples[j+1].docID].has_key(w)):
                            candWordsINDocValue += 1
                        k = k + 1
                        if(k >= max_men_wn):
                            break    
                    if(k == 0):
                        candWordsINDocValue             = 0
                    else:
                        candWordsINDocValue = candWordsINDocValue * 1.0 / k
                elif(self.lang == 'CMN'):
                    for q in range(len(aGroup.samples[j+1].kbNodeStr.strip())):
                        w                               = self.word_vocab.search(aGroup.samples[j+1].kbNodeStr[q])
                        fea[i,j,2 + 4 * max_men_wn + k] = w
                        fea[i,j,2 + 5 * max_men_wn + k] = 1
                        if(self.wordsINDoc[aGroup.samples[j+1].docID].has_key(w)):
                            candWordsINDocValue += 1
                        k = k + 1
                        if(k >= max_men_wn):
                            break    
                    if(k == 0):
                        candWordsINDocValue             = 0
                    else:
                        candWordsINDocValue = candWordsINDocValue * 1.0 / k
        
                #doc match value
                wikiIsNill = ( aGroup.samples[j+1].wikiPageID == '<null>')
                fea[i,j,2 + 6 * max_men_wn] =self.docMatcher.docMatchScore2(self.wikiContextIDs[ aGroup.samples[j+1].wikiPageID], \
                    self.docContextIDs[ aGroup.samples[j+1].docID], \
                    aGroup.samples[j+1].wikiPageID, \
                    aGroup.samples[j+1].docID, \
                    self.doc_max_dis[self.lang], wikiIsNill, self.doc_avg_dis[self.lang])
                
                #string match value
                fea[i,j,3 + 6 * max_men_wn]   = self.stringMatcher.stringMatchScoreLev(aGroup.samples[j+1].headStr, aGroup.samples[j+1].kbNodeStr)
                fea[i,j,4 + 6 * max_men_wn]   = self.stringMatcher.stringMatchScoreLcs(aGroup.samples[j+1].headStr, aGroup.samples[j+1].kbNodeStr)
                fea[i,j,5 + 6 * max_men_wn]   = self.stringMatcher.stringMatchScoreLen(aGroup.samples[j+1].headStr, aGroup.samples[j+1].kbNodeStr)
                fea[i,j,6 + 6 * max_men_wn]   = self.stringMatcher.stringMatchScoreAbbr(aGroup.samples[j+1].headStr, aGroup.samples[j+1].kbNodeStr)
                fea[i,j,7 + 6 * max_men_wn]   = self.stringMatcher.stringMatchScoreAppr(candWordsINDocValue)
                fea[i,j,8 + 6 * max_men_wn]   = self.stringMatcher.stringMatchScoreAppr(menWordsINWikiValue)
                
                #muti names
                if(self.mutiName.has_key(aGroup.samples[j+1].kbNodeID)):
                    for name in self.mutiName[aGroup.samples[j+1].kbNodeID]:
                        ScoreLev = self.stringMatcher.stringMatchScoreLev(aGroup.samples[j+1].headStr, name)
                        ScoreLcs = self.stringMatcher.stringMatchScoreLcs(aGroup.samples[j+1].headStr, name)
                        ScoreLen = self.stringMatcher.stringMatchScoreLen(aGroup.samples[j+1].headStr, name)
                        if(ScoreLev < fea[i,j,3 + 6 * max_men_wn]):
                            fea[i,j,3 + 6 * max_men_wn] = ScoreLev
                        if(ScoreLcs < fea[i,j,4 + 6 * max_men_wn]):
                            fea[i,j,4 + 6 * max_men_wn] = ScoreLcs
                        if(ScoreLen < fea[i,j,5 + 6 * max_men_wn]):
                            fea[i,j,5 + 6 * max_men_wn] = ScoreLen

                #document type
                if(aGroup.samples[j+1].docID.startswith(self.lang + '_NW')):
                    fea[i,j,9 + 6 * max_men_wn] = 0
                elif(aGroup.samples[j+1].docID.startswith(self.lang +'_DF')):
                    fea[i,j,9 + 6 * max_men_wn] = 1
                else:
                    fea[i,j,9 + 6 * max_men_wn] = random.randint(0,1)
        
                #node hot value
                if(self.hotValue.has_key(aGroup.samples[j+1].kbNodeID)):
                    node_hot[j] = self.hotValue[aGroup.samples[j+1].kbNodeID]
                else:
                    node_hot[j] = 0
#                 id = aGroup.samples[j+1].mentionID + aGroup.samples[j+1].kbNodeID
#                 att_words = self.candAttWord[id]
#                 k = 0
#                 for wid in att_words:
#                     fea[i,j,10 + 6 * max_men_wn + k] = wid
#                     fea[i,j,10 + 6 * max_men_wn + k + max_att_wn] = 1
#                     k+=1
#                     if(k >= max_att_wn):
#                         break;
                    
            node_hot  = numpy.floor( node_hot /(  numpy.amax(node_hot) + 0.00000001 )* 9.0 )
            for j in range(len(aGroup.samples)-1):
                fea[i,j,10 + 6 * max_men_wn] = int( node_hot[j] )
                
            lab[i,aGroup.getBoostIdx(), 0] = 1
            #print aGroup.samples[aGroup.getBoostIdx()].toString()
            for j in range(len(aGroup.samples)):
                lab[i,j, 1] = 1
            
        return fea,lab
    
    
    def shuffleGroup(self, aGroup):
        if(len(aGroup.samples) > self.maxSampCount and self.shuffle == 1):
            random.shuffle(aGroup.samples[1:])
            while(len(aGroup.samples) > self.maxSampCount):
                i = 0;
                while(len(aGroup.samples)):
                    if(aGroup.samples[i].kbNodeID == u'NIL' or
                        aGroup.samples[i].rank_value == 1):
                        i += 1
                    else:
                        del(aGroup.samples[i])
                        break;
        return aGroup
    
    def readNextBatch(self):
        count = len(self.samples)
        
        if( count < self.batch_size):
            self.readNextCache()
            
        count = len(self.samples)  
        if( count < self.batch_size):
            if(count == 0):
                return None, None, None
            else:
                while(len(self.samples)<self.batch_size):
                    copy_group = copy.deepcopy(self.samples[-1])
                    self.samples.append(copy_group)
                aBatch = self.samples[0:self.batch_size]
                x, lab = self.fillValue(aBatch)
                del self.samples[0:self.batch_size]
                return aBatch, x, lab
                return aBatch, x, lab

        else:
            aBatch = self.samples[0:self.batch_size]
            x, lab = self.fillValue(aBatch)
            del self.samples[0:self.batch_size]
            return aBatch, x, lab
          
    def readNextCache(self):
        #read file
        i = 0;
        while(i <  self.cache_size and self.FileEnd == 0):
            line = self.file.readline().decode('utf8').strip()
            if(line[0:9] == u"MentionID"):
                if(self.group is not None):
                    if(len(self.group.samples) > 1 ):
                        self.group.samples[0].mentionID   = self.group.samples[1].mentionID
                        self.group.samples[0].mentionType = self.group.samples[1].mentionType
                        self.group.samples[0].headStr     = self.group.samples[1].headStr
                        self.group.samples[0].docID       = self.group.samples[1].docID
                        self.samples.append(self.group)
#                         self.group.printInfoStr()
                        i = i + 1
                        if(i % 1000 == 0):
                            print 'DataProvider.readNextCache.read ', i ,' samples.'
                
                self.group = SampleGroup.SampleGroup()
                self.group.groupName = line[11:]
                continue;
            
            if(not line or line == u'=== doc ==='):
                self.FileEnd = 1
                self.file.close()
                break;
            if(self.group is None):
                break;
            s = Sample(line)
            self.group.samples.append(s)
  
        #shuffle
        if(self.shuffle == 1):
            print 'DataProvider.readNextCache shuffle'
            random.shuffle(self.samples)
        else:  #sort by group size
            print 'DataProvider.readNextCache sort'
            self.samples.sort(key = lambda SampleGroup : len(SampleGroup.samples), reverse = False)

if __name__ == "__main__":
    lang="CMN"
    path="../data/"
    filePath=path+"train.f.txt"
    hotFilePath=path+"NodeHot.txt"
    muti_name_file=path+"multiname.txt"
    att_file=""
    batch_size=8
    cache_size=20000
    maxSampCount=0
    shuffle=1
    word_vocab=Vocab(path+"voc_char.txt")
    word_vocab.load()
    kb_vocab=Vocab(path+"voc_kb.txt")
    kb_vocab.load()
    kbp_type_vocab=Vocab(path+"kb_type.txt")
    kbp_type_vocab.load()
    kb_type_vocab=Vocab(path+"fb_type.txt")
    kb_type_vocab.load()
    dp=DataProvider(lang,  filePath, hotFilePath, muti_name_file, att_file, batch_size, cache_size,
                    maxSampCount, shuffle, word_vocab, kb_vocab, kbp_type_vocab,kb_type_vocab)





