#coding=utf-8
import math

class VocabWord(object):
    def __init__(self, word, cn = 0):
        self.word = word
        self.cn   = cn
    def getWord(self):
        return self.word
    def getCount(self):
        return self.cn
    def toString(self):
        return self.word.encode('utf8') + '\t' + str(self.cn)

class Vocab(object):
    def __init__(self, vocab_file_path):
        self.vocab_file_path = vocab_file_path
        self.__init_params()
    def __init_params(self):
        self.vocab_size  = 0
        self.vocab_words = []
        self.word_map    = {}
    def printVocab(self, name, printSize = -1):
        i = 0;
        print name, ' has total ', self.size(), ' words'; 
        for w in self.vocab_words:
            print str(i) + '\t' + w.toString()
            i = i + 1
            if( printSize > 0 and i > printSize):
                break
        
    def addWord(self, vocabWord):
        if(not self.word_map.has_key(vocabWord.getWord())):
            self.word_map[vocabWord.getWord()] = self.vocab_size;
            self.vocab_words.append(vocabWord)
            self.vocab_size = self.vocab_size + 1
    def size(self):
        return self.vocab_size
    def load(self):
        lines = open(self.vocab_file_path, "r")
        self.__init_params()
        for line in lines:
            line = line.decode('utf8').strip()
            info = line.split('\t');
            if(len(info) != 2):
                #print 'Vocab.load warning:', line.encode('utf8')
                continue
            self.addWord(VocabWord(info[0],info[1]))
        self.unkID = self.search(u'%UNK%')
    
    def exists(self, word):
        if(self.word_map.has_key(word)):
            return True
        else:
            return False
    
    def search(self, word):
        id = -1
        if(self.word_map.has_key(word)):
            id = self.word_map[word]
        else:
            id = self.unkID;
        return id

if __name__ == '__main__':
    vocab = Vocab('vocab.txt');
    vocab.addWord(VocabWord('aa', 1))
    vocab.addWord(VocabWord('bb', 2))
    vocab.addWord(VocabWord('cc', 3))
    #vocab.load()
    print 'vocab size:', vocab.size()
    
    idx = vocab.search('aa')
    print idx
    
    idx = vocab.search('4')
    print idx
