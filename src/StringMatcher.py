
# -*- coding:utf-8 -*-
import re

class StringMatcher(object):

    def __init__(self):
        self.pattern_abbr = re.compile(r'^[A-Z\.]+$')
        self.cache_lev        = {}
        self.cache_score_lev  = {}
        self.cache_score_lcs  = {}
        self.cache_score_abbr = {}
        self.cache_max_size   = 100000
    
    def levenshtein(self,A,B):
        key_str = A + ' +=+ ' + B
        if(self.cache_lev.has_key(key_str)):
            return self.cache_lev[key_str]

        len_A = len(A) + 1
        len_B = len(B) + 1
        if len_A == 1:
            return len_B -1
        if len_B == 1:
            return len_A -1
        matrix = [range(len_A) for x in range(len_B)]
        for i in range(1,len_B):
            matrix[i][0] = i
        for i in range(1,len_B):
            for j in range(1,len_A):
                deletion = matrix[i-1][j]+1
                insertion = matrix[i][j-1]+1
                substitution = matrix[i-1][j-1]
                if B[i-1] != A[j-1]:
                    substitution += 1
                    matrix[i][j] = min(deletion,insertion,substitution)
                    #print matrix
        
        self.cache_lev[key_str] = matrix[len_B-1][len_A-1]
        if(len(self.cache_lev) > self.cache_max_size):
            self.cache_lev.clear()
        return matrix[len_B-1][len_A-1]
    
    def isAbbr(self, s):
        if(self.pattern_abbr.match(s)):
            return 1
        else:
            return 0
    
    def getCap(self, str):
        cap = "";
        for s in str.split():
            cap += s[0]
        return cap
    
        
    def find_lcs_len(self, s1, s2): 
        m = [ [ 0 for x in s2 ] for y in s1 ] 
        for p1 in range(len(s1)): 
            for p2 in range(len(s2)): 
                if s1[p1] == s2[p2]: 
                    if p1 == 0 or p2 == 0: 
                        m[p1][p2] = 1
                    else: 
                        m[p1][p2] = m[p1-1][p2-1]+1
                elif m[p1-1][p2] < m[p1][p2-1]: 
                    m[p1][p2] = m[p1][p2-1] 
                else:               # m[p1][p2-1] < m[p1-1][p2] 
                    m[p1][p2] = m[p1-1][p2] 
        return m[-1][-1]        
     
    
    def stringMatchScoreLev(self, str1, str2):
        str1    = str1.lower()
        str2    = str2.lower()
        key_str = str1 + ' +=+ ' + str2
        if(self.cache_score_lev.has_key(key_str)):
            return self.cache_score_lev[key_str]

        rlt = 5
        if(str1 == str2):
            rlt = 0
        else:
            dis   = self.levenshtein(str1, str2)
            if(dis <= 1):
                rlt = 1
            elif(dis <= 2):
                rlt = 2
            elif(dis <= 3):
                rlt = 3
            elif(dis <= 4):
                rlt = 4
            else:
                rlt = 5
        self.cache_score_lev[key_str] = rlt
        if(len(self.cache_score_lev) > self.cache_max_size):
            self.cache_score_lev.clear()
        return rlt
    
    def stringMatchScoreLcs(self, str1, str2):
        str1 = str1.lower()
        str2 = str2.lower()
        key_str = str1 + ' +=+ ' + str2
        if(self.cache_score_lcs.has_key(key_str)):
            return self.cache_score_lcs[key_str]

        rlt = 5
        if(str1 == str2):
            rlt = 0
        else:
            if(len(str1) == 0 or len(str2) == 0):
                rlt = 5
            else:
                dis   = self.find_lcs_len(str1, str2)
                short__ = min(len(str1), len(str2)) 
                r     = dis*1.0/short__
                
                if(r >= 0.9):
                    rlt = 1
                elif(r >= 0.7):
                    rlt = 2
                elif(r >= 0.5):
                    rlt = 3
                elif(r >= 0.3):
                    rlt = 4
                else:
                    rlt = 5
        self.cache_score_lcs[key_str] = rlt
        if(len(self.cache_score_lcs) > self.cache_max_size):
            self.cache_score_lcs.clear()
        return rlt
    
    def stringMatchScoreLen(self, str1, str2):
        l1    = len(str1)
        l2    = len(str2)
        if(l1 == l2):
            return 0
        elif(l1 > l2):
            r = l2 * 1.0 / l1
            if( r >= 0.8):
                return 1
            elif(r >= 0.6):
                return 3
            else:
                return 5
        else:
            r = l1 * 1.0 / l2
            if( r >= 0.8):
                return 2
            elif(r >= 0.6):
                return 4
            else:
                return 5
    
    def stringMatchScoreAbbr(self, s1, s2): 
        key_str = s1 + ' +=+ ' + s2
        if(self.cache_score_abbr.has_key(key_str)):
            return self.cache_score_abbr[key_str]

        rlt = 5
        if(self.isAbbr(s1) == 1 and self.isAbbr(s2) == 0):
            s1  = s1.lower().replace('.','')
            cap = self.getCap(s2).lower()
            if(s1 == cap):
                rlt = 0
            else:
                lev = self.levenshtein(s1, cap)
                if(lev <= 1):
                    rlt = 1
                elif(lev <= 2):
                    rlt = 2
                elif(lev <= 3):
                    rlt = 3
                else:
                    rlt = 4
        elif(self.isAbbr(s1) == 0 and self.isAbbr(s2) == 1):
            s2  = s2.lower().replace('.','')
            cap = self.getCap(s1).lower()
            if(s2 == cap):
                rlt = 0
            else:
                lev = self.levenshtein(s2, cap)
                if(lev <= 1):
                    rlt = 1
                elif(lev <= 2):
                    rlt = 2
                elif(lev <= 3):
                    rlt = 3
                else:
                    rlt = 4
            
        elif(self.isAbbr(s1) == 1 and self.isAbbr(s2) == 1):
            s1 = s1.lower().replace('.','')
            s2 = s2.lower().replace('.','')
            if(s1 == s2):
                rlt = 0
            else:
                lev = self.levenshtein(s1, s2)
                if(lev <= 1):
                    rlt = 1
                elif(lev <= 2):
                    rlt = 2
                elif(lev <= 3):
                    rlt = 3
                else:
                    rlt = 4
        else:
            rlt = 5         
        self.cache_score_abbr[key_str] = rlt
        if(len(self.cache_score_abbr) > self.cache_max_size):
            self.cache_score_abbr.clear()
        return rlt


    def stringMatchScoreAppr(self, otherValue1):
        r     = otherValue1
        if(r == 1):
            return 0
        elif(r >= 0.9):
            return 1
        elif(r >= 0.7):
            return 2
        elif(r >= 0.5):
            return 3
        elif(r >= 0.3):
            return 4
        else:
            return 5



if __name__ == '__main__':
    s1 = u'浙江大学'
    s2 = u'浙大'
    sm = StringMatcher()
    print s1
    print s2
    print '==================='
    print sm.levenshtein(s1,s2);
    print '==================='
    print sm.find_lcs_len(s1, s2)
    print '==================='

    
    