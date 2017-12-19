#coding=utf-8
'''
Created on Mar 14, 2016

@author: weilin2
'''

import numpy
import theano
import theano.tensor as TT
import time
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from Vocab import *
from Cfg import *
from Sample import *
from Optimiser import *
from DataProvider import *
from ultils import *
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


max_men_wn = 10                 #mention word number
max_doc_wn = 800                #document word number
sen_word_cn = 20                #sentence word count

class NNRank(object):
    '''
    rank lib for KBP EDL task
    '''

    def __init__(self, cfg, lang = 'ENG', model_file = None, test_file = None):
        
        self.lang            = lang
        self.cfg             = cfg;
        self.word_emb_file   = cfg.state['word_emb_file']
        self.word_emb_size   = cfg.state['word_emb_size']
        self.word_vocab_file = cfg.state['word_vocab_file']
        
        self.kb_emb_file     = cfg.state['kb_emb_file']
        self.kb_emb_size     = cfg.state['kb_emb_size']
        self.kb_vocab_file   = cfg.state['kb_vocab_file']
        
        self.kbp_type_vocab_file = cfg.state['kbp_type_vocab_file']
        self.kbp_type_emb_size   = cfg.state['kbp_type_emb_size']
        self.fb_type_vocab_file  = cfg.state['fb_type_vocab_file']
        self.fb_type_emb_size    = cfg.state['fb_type_emb_size']
        
        self.stringMatchVocabSzie = cfg.state['string_match_vocab_size']
        self.stringMatchEmbSize   = cfg.state['string_match_emb_size']
        self.docMatchEmbSize      = 10
        self.word_emb_file        = cfg.state['word_emb_file'] 

        self.hid1_size       = cfg.state['hid1_size']
        self.hid2_size       = cfg.state['hid2_size']
        self.hid1_size_NIL   = 128
        self.hid2_size_NIL   = 128
        self.init_lr         = cfg.state['init_lr']
        self.weight_decay    = float(cfg.state['weight_decay'])
        self.batch_size      = cfg.state['batch_size']
        self.epoc            = cfg.state['epoc']
        self.declr_start     = cfg.state['declr_start']
        
        self.static_emb      = cfg.state['static_emb']
        self.dropout         = 0.0
        
        self.outdir          = cfg.state['outdir']
        self.trainFile       = cfg.state['train']
        self.validFile       = cfg.state['valid']
        self.hotFile         = cfg.state['node_hot']
        self.mutiNameFile    = cfg.state['object_muti_name']
        self.testFile        = cfg.state['test']
        self.trainAttFile    = cfg.state['trainAttFile']
        self.testAttFile     = cfg.state['testAttFile']
        self.modelFile       = cfg.state['model']
        self.ran_seed        = cfg.state['ran_seed']
        self.cache_size      = cfg.state['cache_size']
        self.debug_mod       = cfg.state['debug_model']

        if(model_file is not None):
            self.modelFile   = model_file
        if(test_file is not None):
            self.testFile    = test_file
		        
        self.docTypeVocabSize = 3
        self.docTypeEmbSize   = 5
        self.nodeHotEmbSize   = 10
        
        self.emb_size        = self.word_emb_size * 2 + self.docMatchEmbSize + \
            self.kbp_type_emb_size + self.fb_type_emb_size + self.stringMatchEmbSize * 6 + self.docTypeEmbSize + 10
        
        self.emb_size2       = self.word_emb_size + self.kbp_type_emb_size + self.docTypeEmbSize
         
        
        self.tparams         = []

        numpy.random.seed(self.ran_seed)
        self.rng = numpy.random.RandomState(self.ran_seed + 1)
        #self.trng = RandomStreams(self.rng.randint(42545))
        
        self.loadVocabs()
        self.__init_all_matrix()
        if(os.path.exists(self.word_emb_file)):
            self.loadEmb(self.word_emb_file, self.word_emb_size, self.matirx_word_emb , self.word_vocab)
            
        if(os.path.exists(self.modelFile + '.npz')):
            self.loadModel(self.modelFile)
        self.optimiser = AdaDelta(self.tparams)
        
    
    def loadVocab(self, vocab_file, vocab_name):
        vob = Vocab(vocab_file)
        vob.load()
        if(self.debug_mod >= 1):
            vob.printVocab(vocab_name, 10)
        return vob
    
    def loadVocabs(self):
        self.word_vocab     = self.loadVocab(self.word_vocab_file    , 'word_vocab')
        self.kb_vocab       = self.loadVocab(self.kb_vocab_file      , 'kb_vocab')
        self.kbp_type_vocab = self.loadVocab(self.kbp_type_vocab_file, 'kbp_type_vocab')
        self.fb_type_vocab  = self.loadVocab(self.fb_type_vocab_file , 'fb_type_vocab')
    
    def __init_matrix(self, shape, matrix_name):
        matrix = theano.shared(value = \
                 numpy.asarray(self.rng.uniform(low=-0.1, high=0.1, size= shape),\
                                dtype=theano.config.floatX),\
                 name = matrix_name,\
                 borrow = True )
        if(self.debug_mod >= 2):
            print matrix_name
            print matrix.get_value()
        return matrix
    
    
    def __init_all_matrix(self):  
        if(self.debug_mod >= 1):
            print '__init_Matrix begin'
        
        self.matirx_word_emb        =  self.__init_matrix((self.word_vocab.size(), self.word_emb_size),         'matirx_word_emb')
        self.matrix_kbp_type_emb    =  self.__init_matrix((self.kbp_type_vocab.size(), self.kbp_type_emb_size), 'matrix_kbp_type_emb')
        self.matrix_fb_type_emb     =  self.__init_matrix((self.fb_type_vocab.size(), self.fb_type_emb_size),   'matrix_fb_type_emb')
        self.matrix_doc_match_emb   =  self.__init_matrix((10, self.docMatchEmbSize),                           'matrix_doc_match_emb')
        self.matrix_str_match_emb1  =  self.__init_matrix((self.stringMatchVocabSzie, self.stringMatchEmbSize), 'matrix_str_match_emb1')
        self.matrix_str_match_emb2  =  self.__init_matrix((self.stringMatchVocabSzie, self.stringMatchEmbSize), 'matrix_str_match_emb2')
        self.matrix_str_match_emb3  =  self.__init_matrix((self.stringMatchVocabSzie, self.stringMatchEmbSize), 'matrix_str_match_emb3')
        self.matrix_str_match_emb4  =  self.__init_matrix((self.stringMatchVocabSzie, self.stringMatchEmbSize), 'matrix_str_match_emb4')
        self.matrix_str_match_emb5  =  self.__init_matrix((self.stringMatchVocabSzie, self.stringMatchEmbSize), 'matrix_str_match_emb5')
        self.matrix_str_match_emb6  =  self.__init_matrix((self.stringMatchVocabSzie, self.stringMatchEmbSize), 'matrix_str_match_emb6')
        self.matrix_doc_type_emb    =  self.__init_matrix((self.docTypeVocabSize, self.docTypeEmbSize),         'matrix_doc_type_emb')
        self.matrix_node_hot_emb    =  self.__init_matrix((10, 10),                                             'matrix_node_hot_emb')
        self.matrix_emb_h1          =  self.__init_matrix((self.emb_size  , self.hid1_size),                    'matrix_emb_h1')
        self.matrix_h1_h2           =  self.__init_matrix((self.hid1_size , self.hid2_size),                    'matrix_h1_h2')
        self.matrix_h2_out          =  self.__init_matrix((self.hid2_size , 1),                                 'matrix_h2_out')
        self.matrix_emb_h1_NIL      =  self.__init_matrix(( self.emb_size2, self.hid1_size_NIL),                    'matrix_emb_h1_NIL')
        self.matrix_h1_h2_NIL       =  self.__init_matrix((self.hid1_size_NIL , self.hid2_size_NIL),                    'matrix_h1_h2_NIL')
        self.matrix_h2_out_NIL      =  self.__init_matrix((self.hid2_size_NIL , 1),                                 'matrix_h2_out_NIL')

        self.tparams.append(self.matirx_word_emb)
        self.tparams.append(self.matrix_kbp_type_emb)
        self.tparams.append(self.matrix_fb_type_emb)
        self.tparams.append(self.matrix_doc_match_emb)
        self.tparams.append(self.matrix_str_match_emb1)
        self.tparams.append(self.matrix_str_match_emb2)
        self.tparams.append(self.matrix_str_match_emb3)
        self.tparams.append(self.matrix_str_match_emb4)
        self.tparams.append(self.matrix_str_match_emb5)
        self.tparams.append(self.matrix_str_match_emb6)
        self.tparams.append(self.matrix_doc_type_emb)
        self.tparams.append(self.matrix_node_hot_emb)
        self.tparams.append(self.matrix_emb_h1)
        self.tparams.append(self.matrix_h1_h2)
        self.tparams.append(self.matrix_h2_out)
        self.tparams.append(self.matrix_emb_h1_NIL)
        self.tparams.append(self.matrix_h1_h2_NIL)
        self.tparams.append(self.matrix_h2_out_NIL)
    
    
    def _avgEmbs(self, matrix, x_input, x_mask, bSize, sc, wn, emb_size):
        emb  = matrix[x_input.flatten()]
        emb  = emb.reshape((bSize, sc, wn, emb_size))
        emb  = emb * (TT.cast(x_mask, dtype = 'float32').dimshuffle(0,1,2,'x'))
        emb  = (emb.sum(axis = 2) / (TT.cast(x_mask.sum(axis = 2) , dtype = 'float32') + TT.cast(1e-8, dtype = 'float32')).dimshuffle(0, 1,'x'))
        return emb
    
    def _embs(self, matrix, x_input):
        return matrix[x_input.flatten()]
    
    # def _dropout(self,x, x_shape, trainMod):
    #     if(trainMod == 1):
    #         x  = x * self.trng.binomial(x_shape, p = (1 - self.dropout), dtype = theano.config.floatX)
    #     else:
    #         x  = x * (1 - self.dropout)
    #     return x
    
    def _buildPairModel(self, fea, trainMod):
        relu_func = eval('ReLU')
        #fea ==> (bSize, max_sample_count-1, dim)
        
        #kbNode_type_emb ==> (bSize, max_sample_count-1, emb_dim)
        kbNode_type_emb  = self._avgEmbs(self.matrix_fb_type_emb,\
                                         fea[:,:,2 + 0 * max_men_wn : 2 + 1 * max_men_wn],\
                                         fea[:,:,2 + 1 * max_men_wn : 2 + 2 * max_men_wn],\
                                         self.batch_size, fea.shape[1], max_men_wn, self.fb_type_emb_size)

        mention_str_emb  = self._avgEmbs(self.matirx_word_emb,\
                                         fea[:,:,2 + 2 * max_men_wn : 2 + 3 * max_men_wn],\
                                         fea[:,:,2 + 3 * max_men_wn : 2 + 4 * max_men_wn],\
                                         self.batch_size,fea.shape[1], max_men_wn, self.word_emb_size)
        kbNode_str_emb   = self._avgEmbs(self.matirx_word_emb,\
                                         fea[:,:,2 + 4 * max_men_wn : 2 + 5 * max_men_wn],\
                                         fea[:,:,2 + 5 * max_men_wn : 2 + 6 * max_men_wn],\
                                         self.batch_size,fea.shape[1], max_men_wn, self.word_emb_size)
        #mention_type_emb ==> (bSize, max_sample_count-1, emb_dim)
        
        mention_type_emb  = self._embs(self.matrix_kbp_type_emb   , fea[:,:,0]).reshape((self.batch_size, fea.shape[1], self.kbp_type_emb_size))
        doc_match_emb     = self._embs(self.matrix_doc_match_emb  , fea[:,:,2 + 6 * max_men_wn]).reshape((self.batch_size, fea.shape[1], self.docMatchEmbSize))
        str_match_emb1    = self._embs(self.matrix_str_match_emb1 , fea[:,:,3 + 6 * max_men_wn]).reshape((self.batch_size, fea.shape[1], self.stringMatchEmbSize))
        str_match_emb2    = self._embs(self.matrix_str_match_emb2 , fea[:,:,4 + 6 * max_men_wn]).reshape((self.batch_size, fea.shape[1], self.stringMatchEmbSize))
        str_match_emb3    = self._embs(self.matrix_str_match_emb3 , fea[:,:,5 + 6 * max_men_wn]).reshape((self.batch_size, fea.shape[1], self.stringMatchEmbSize))
        str_match_emb4    = self._embs(self.matrix_str_match_emb4 , fea[:,:,6 + 6 * max_men_wn]).reshape((self.batch_size, fea.shape[1], self.stringMatchEmbSize))
        str_match_emb5    = self._embs(self.matrix_str_match_emb5 , fea[:,:,7 + 6 * max_men_wn]).reshape((self.batch_size, fea.shape[1], self.stringMatchEmbSize))
        str_match_emb6    = self._embs(self.matrix_str_match_emb6 , fea[:,:,8 + 6 * max_men_wn]).reshape((self.batch_size, fea.shape[1], self.stringMatchEmbSize))
        doc_type_emb      = self._embs(self.matrix_doc_type_emb   , fea[:,:,9 + 6 * max_men_wn]).reshape((self.batch_size, fea.shape[1], self.docTypeEmbSize))
        node_hot_emb      = self._embs(self.matrix_node_hot_emb   , fea[:,:,10 + 6 * max_men_wn]).reshape((self.batch_size, fea.shape[1], self.nodeHotEmbSize))

#         att_word_emb      = self._avgEmbs(self.matirx_word_emb,\
#                                          fea[:,:,10 + 6 * max_men_wn + 0 * max_att_wn : 10 + 6 * max_men_wn + 1 * max_att_wn ],\
#                                          fea[:,:,10 + 6 * max_men_wn + 1 * max_att_wn : 10 + 6 * max_men_wn + 2 * max_att_wn],\
#                                          self.batch_size,fea.shape[1], max_att_wn, self.word_emb_size)

        #mention_type_emb ==> (bSize, max_sample_count-1, emb_dim)
#         emb   = TT.concatenate([mention_str_emb, mention_type_emb, kbNode_emb, kbNode_str_emb, \
#                                kbNode_type_emb, doc_match_emb, str_match_emb1, str_match_emb2, str_match_emb3, \
#                                str_match_emb4, str_match_emb5, str_match_emb6 ,doc_type_emb, node_hot_emb, \
#                                att_word_emb ], axis = 2)
        emb   = TT.concatenate([mention_str_emb, mention_type_emb, kbNode_str_emb, \
                               kbNode_type_emb, doc_match_emb, str_match_emb1, str_match_emb2, str_match_emb3, \
                               str_match_emb4, str_match_emb5, str_match_emb6 ,doc_type_emb, node_hot_emb\
                             ], axis = 2)
        
        #emb = self._dropout(emb, (self.batch_size, self.maxSampCount-1, self.emb_size), trainMod)
        
        #hid1 ==> (bSize, max_sample_count-1, hid1)
        hid1  = relu_func( TT.dot(emb, self.matrix_emb_h1))
        #hid1  = self._dropout(hid1, (self.batch_size, self.maxSampCount-1, self.hid1_size), trainMod)
        hid2  = relu_func( TT.dot(hid1, self.matrix_h1_h2))
        #hid2  = self._dropout(hid2, (self.batch_size, self.maxSampCount-1, self.hid2_size), trainMod)
        #hid1 ==> (bSize, max_sample_count-1, 1)

        out   = TT.dot(hid2, self.matrix_h2_out)
        return out   
    
    
    def _buildNilModel(self, fea, trainMod):
        relu_func = eval('ReLU')
        #fea ==> (bSize, max_sample_count-1, dim)
        
        #mention_str_emb ==> (bSize, 1, emb_dim)
        mention_str_emb  = self._avgEmbs(self.matirx_word_emb,\
                                         fea[:,0,2 + 2 * max_men_wn : 2 + 3 * max_men_wn].reshape((self.batch_size, 1, max_men_wn)),\
                                         fea[:,0,2 + 3 * max_men_wn : 2 + 4 * max_men_wn].reshape((self.batch_size, 1, max_men_wn)),\
                                         self.batch_size, 1, max_men_wn, self.word_emb_size)
        

        mention_type_emb  = self._embs(self.matrix_kbp_type_emb , fea[:,0,0]).reshape((self.batch_size, 1, self.kbp_type_emb_size))
        doc_type_emb      = self._embs(self.matrix_doc_type_emb , fea[:,0,9 + 6 * max_men_wn]).reshape((self.batch_size, 1, self.docTypeEmbSize))

        
        emb   = TT.concatenate([mention_str_emb, mention_type_emb,doc_type_emb], axis = 2)
        # emb = self._dropout(emb, (self.batch_size, 1, self.emb_size2), trainMod)
        
        hid1  = relu_func( TT.dot(emb, self.matrix_emb_h1_NIL))
        # hid1  = self._dropout(hid1, (self.batch_size, 1, self.hid1_size_NIL), trainMod)
        hid2  = relu_func( TT.dot(hid1, self.matrix_h1_h2_NIL))
        # hid2  = self._dropout(hid2, (self.batch_size, 1, self.hid2_size_NIL), trainMod)
        #hid1 ==> (bSize, 1, hid1)
        
        out   = TT.dot(hid2, self.matrix_h2_out_NIL)
        
        return out   
        
        
    def train_functions(self, trainMod):

        fea      = TT.tensor3(name = 'fea'   , dtype = 'int32')
        lab      = TT.tensor3(name = 'lab'   , dtype = 'int32')
        
        out0 = self._buildNilModel(fea, trainMod)
        out1 = self._buildPairModel(fea, trainMod)
        
        #softmax
        out     = TT.concatenate([out0, out1], axis = 1).reshape((self.batch_size, lab.shape[1]))
        out_max = out.max(axis = 1) # (bSize)
        out_exp = TT.exp(out - out_max.dimshuffle(0,'x')) * TT.cast(lab[:,:,1], dtype = 'float32') + TT.cast(1e-8, dtype = 'float32')
        out_sum = out_exp.sum(axis = 1)
        probs   = out_exp / out_sum.dimshuffle(0,'x')
        
        cost     = TT.nnet.categorical_crossentropy(probs, TT.cast(lab[:,:,0], dtype = 'float32')).mean()
        grads = TT.grad(cost, self.tparams )
        return self.optimiser.Optimiser(self.tparams, grads, [fea, lab] , cost)
    
    

    def test_functions(self, trainMod):
        fea      = TT.tensor3(name = 'fea'   , dtype = 'int32')
        lab      = TT.tensor3(name = 'lab'   , dtype = 'int32')

        out0 = self._buildNilModel( fea, trainMod)
        out1 = self._buildPairModel(fea, trainMod)
        
        out     = TT.concatenate([out0, out1], axis = 1).reshape((self.batch_size, lab.shape[1]))
        out_max = out.max(axis = 1) # (bSize)
        out_exp = TT.exp(out - out_max.dimshuffle(0,'x')) * TT.cast(lab[:,:,1], dtype = 'float32') + TT.cast(1e-8, dtype = 'float32')
        out_sum = out_exp.sum(axis = 1)
        probs   = out_exp / out_sum.dimshuffle(0,'x')
        #probs = dbg_hook_shape('probs', probs)

        ftest = theano.function([fea,lab], probs, allow_input_downcast = True,on_unused_input = 'ignore')
        
        return ftest

    def train(self):
        iter = 0
        lr   = float(self.init_lr)
        (fcost , fupdata ) = self.train_functions(1)
        
        trainDataProvider = DataProvider(self.lang, self.trainFile, self.hotFile, self.mutiNameFile, self.trainAttFile, self.batch_size, self.cache_size, 0, 1,  \
                            self.word_vocab, self.kb_vocab, self.kbp_type_vocab, self.fb_type_vocab)

        while(iter < self.epoc):
            cost  = 0;
            count = 0
            trainDataProvider.reset()
            beginTime = time.time()
            while(trainDataProvider.isEnd() == 0):
                aBatch, fea, lab = trainDataProvider.readNextBatch()
                # for aGroup in aBatch:
                #     for s in aGroup.samples:
                #         print s.toString()
                # print fea
                # print lab
                
                tmp = self.processOneBatch(aBatch, fea, lab, lr, fcost , fupdata)
                if(tmp >= 0):
                    cost = cost + tmp
                    count = count + self.batch_size
                    if(count/ self.batch_size % 10 == 0 and count > 0):
                        fTime = time.time() - beginTime
                        print "iter: %d, lr: %.6f, processed count: %d , avg cost: %.6f, speed: %.2f ins/s" \
                            %(iter, lr, count, cost / count,count/ fTime )
            self.saveModel(iter)  
            iter = iter + 1
            if(iter >= self.declr_start):
                lr = lr / 2
    
    def test(self):
        ftest = self.test_functions(0)
        testDataProvider = DataProvider(self.lang, self.testFile, self.hotFile, self.mutiNameFile, self.testAttFile,  self.batch_size, self.cache_size, 0 , 0, \
                            self.word_vocab, self.kb_vocab, self.kbp_type_vocab, self.fb_type_vocab)
        testDataProvider.reset()
        count = 0;
        right = 0;
        while(testDataProvider.isEnd() == 0):
            aBatch, fea, lab = testDataProvider.readNextBatch()
            rlt = self.testOneBatch(aBatch, fea, lab, ftest)
            if(rlt == 1):
                right = right + 1
            count = count + 1
        print 'total: %d groups, right %d groups (%.2f %%)' %(count, right, right * 100.0 / count)    
        
    def testOneBatch(self, aBatch, fea, lab, ftest):
        if(aBatch is None):
            print 'aBatchSample is None'
            return

        out = ftest(fea, lab )
          
        for i in range(self.batch_size):
            aGroup = aBatch[i]
            print "MentionID:",aGroup.groupName
            tmp = aGroup.samples
      
            for j in range(len(tmp)):
                tmp[j].predictScore = out[i][j]
            tmp.sort(key = lambda Sample : Sample.predictScore, reverse = True)
             
            j = 0;
            for s in tmp:
                print s.predictScore, s.toString()
                j = j + 1;
                if(j >= 10):
                    break
        return 0
        
        
                
    def processOneBatch(self, aBatch, fea, lab,  lr, fcost, fupdata):
        if(aBatch is None):
            print 'aBatch is None'
            return -1
        
        cost = fcost(fea, lab)
        fupdata(lr)
        return cost
    
    def saveModel(self, iter):
        parms = OrderedDict()
        for tp in self.tparams:
            parms[tp.name] = tp.get_value()
        numpy.savez(self.modelFile + "_iter" + str(iter), **parms) 
        
    def loadModel(self, modelFile):
        print 'load model from ', modelFile
        parms = numpy.load(modelFile + '.npz')
        parm_dict= {}
        for k in parms.keys():
            parm_dict[k] = parms[k]
        for tp in self.tparams:
            if not parm_dict.has_key(tp.name):
                print 'model parm for %s expected' %tp.name
                continue 
            tp.set_value(parm_dict[tp.name]) 
    
    def loadEmb(self, emb_file, emb_size, tensorMatrix, vocab):
        print 'load embedding from', emb_file
        file = open(emb_file, 'rb')
        dim = int(file.readline().decode('utf8').strip().split(' ')[-1])
        matrix     = tensorMatrix.get_value()
        assert(dim == emb_size)
        loaded     = 0
        vocab_size = 0
        while(1):
            line = file.readline().decode('utf8').strip()
            if(line):
                line = line.replace('\t',' ')
                array = line.split(' ')
                if( (dim + 1) != len(array)):
                    #print 'loadEmb warning:', line.encode('utf8')
                    continue
                assert(dim + 1 == len(array))
                w = array.pop(0)
                if(vocab.word_map.has_key(w)):
                    id = vocab.search(w)
                    for i in xrange(len(array)):
                        matrix[id, i] = float(array[i])
                    loaded = loaded + 1
                vocab_size += 1
            else:
                break; 
        file.close()
        tensorMatrix.set_value(matrix)
        print 'load embedding finished, total %d words %d with the pre embedding' % (vocab_size, loaded)

if __name__ == '__main__':
    print 'NNRank.main start'
    # numpy.set_printoptions(threshold='nan')
    # cfgFilePath = sys.argv[1]
    # process     = sys.argv[2]
    # lang        = sys.argv[3]
    # model_file  = None
    # test_file   = None
    # if(len(sys.argv) >= 6 ):
    #     model_file  = sys.argv[4]
    #     test_file   = sys.argv[5]
    cfgFilePath = "cfg.txt"
    process= "train"
    lang= "CMN"
    model_file  = None
    test_file   = None
    cfg    = Cfg(cfgFilePath)
    nnRank = NNRank(cfg, lang, model_file, test_file)
    
    if(process == "train"):
        print "NNRank train."
        if(os.path.exists(nnRank.trainFile)):
            nnRank.train()
     
    if(process == "test"):  
        print "NNRank test."
        if(os.path.exists(nnRank.testFile)):
            nnRank.test()