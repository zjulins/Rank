# -*- coding:utf-8 -*-
'''
Created on Mar 31, 2016

@author: weilin2
'''

import math
from Vocab import *
class DocMatcher(object):

    def __init__(self):
        self.idf          = {}
        self.cache_score  = {}
        self.cache_score2 = {}
        self.unk_idf      = 0
        self.count        = 0
        self.dis_sum      = 0
        self.cache_max_size   = 1000000
        
    def _loadIDF(self, idf_file_path, word_vocab):
        file = open(idf_file_path, 'rb')
        loaded = 0
        count  = 0
        for line in file.readlines():
            try:
                line = line.decode('utf8').strip()
                term = line.split('\t')
                if(word_vocab.exists(term[0])):
                    w_id = word_vocab.search(term[0])
                    self.idf[w_id] = float(term[1])
                    loaded += 1
            except:
                pass
            count += 1
        self.unk_idf = self.idf[word_vocab.unkID]
        print 'DocMatcher._loadIDF end. total %d lines, load %d lines' %(count, loaded)
        file.close()
    
    def cos_dis(self, v1, v2):
        t = {}
        c = 0
        for v in v1:
            if(not t.has_key(v)):
                t[v] = c
                c += 1
        for v in v2:
            if(not t.has_key(v)):
                t[v] = c
                c += 1
        
        vector1 = [0] * c
        vector2 = [0] * c
        
        for v in v1:
            vector1[t[v]] = 1
        for v in v2:
            vector2[t[v]] = 1
            
        dot  = 0
        nom1 = 0
        nom2 = 0
        for i in range(c):
            dot  += vector1[i] * vector2[i]
            nom1 += vector1[i] * vector1[i]
            nom2 += vector2[i] * vector2[i]
        
        fm = math.sqrt(nom1) * math.sqrt(nom2)
        if(fm == 0):
            dis = 0
        else:    
            dis = dot / fm
        
        return dis
    
    def tfidf_cos_dis(self, v1, v2):
        t = {}
        c = 0
        tf1 = {}
        tf2 = {}
        
        for v in v1:
            if(tf1.has_key(v)):
                tf1[v] += 1
            else:
                tf1[v] = 1
            if(not t.has_key(v)):
                t[v] = c
                c += 1
        for v in v2:
            if(tf2.has_key(v)):
                tf2[v] += 1
            else:
                tf2[v] = 1
            if(not t.has_key(v)):
                t[v] = c
                c += 1
        
        vector1 = [0] * c
        vector2 = [0] * c
        
        for v in v1:
            if(self.idf.has_key(v)):
                idf = self.idf[v]
            else:
                idf = self.unk_idf
            if idf>8.5:
                idf=0
            vector1[t[v]] = tf1[v]* idf
            
        for v in v2:
            if(self.idf.has_key(v)):
                idf = self.idf[v]
            else:
                idf = self.unk_idf
            if idf>8.5:
                idf=0
            vector2[t[v]] = tf2[v]* idf
            
        dot  = 0
        nom1 = 0
        nom2 = 0
        for i in range(c):
            dot  += vector1[i] * vector2[i]
            nom1 += vector1[i] * vector1[i]
            nom2 += vector2[i] * vector2[i]
        
        fm = math.sqrt(nom1) * math.sqrt(nom2)
        if(fm == 0):
            dis = 0
        else:    
            dis = dot / fm
        
        return dis
    
    def docMatchScore(self,v1, v2, doc_id1, doc_id2):
        key_str = doc_id1 + ' +=+ ' + doc_id2
        if(self.cache_score.has_key(key_str)):
            return self.cache_score[key_str]

        dis = self.cos_dis(v1, v2)
        dis = int((dis - 0.000001) * 10)

        self.cache_score[key_str] = dis
        if(len(self.cache_score) > self.cache_max_size):
            self.cache_score.clear()
        return dis #return 0~9
        
    def docMatchScore2(self,v1, v2, doc_id1, doc_id2, maxDis = 1,  wikiIsNull = False, avgDis = 0):
        key_str = doc_id1 + ' +=+ ' + doc_id2
        if(self.cache_score2.has_key(key_str)):
            return self.cache_score2[key_str]

        dis = self.tfidf_cos_dis(v1, v2)
        if(wikiIsNull == True):
            dis = avgDis
        if(dis > maxDis):
            dis = maxDis
        dis = dis / maxDis
        dis = int((dis - 0.000001) * 10)

        self.cache_score2[key_str] = dis
        if(len(self.cache_score2) > self.cache_max_size):
            self.cache_score2.clear()

        return dis #return 0~9
        
if __name__ == "__main__":
    vocab=Vocab("../data/voc_char.txt")
    vocab.load()
    mat=DocMatcher()
    mat._loadIDF("../data/IDF.txt",vocab)
    v1=['a','f','c','%UNK%','%UNK%']
    v2=['a','b','e','askasasas','aaaaaaasadad']
    v3=[]
    v4=[]
    for v in v1:
        v3.append(vocab.search(v))
    for v in v2:
        v4.append(vocab.search(v))
    print mat.tfidf_cos_dis(v3,v4)