'''
Created on Mar 14, 2016

@author: weilin2
'''

import os

class Cfg(object):
    def __init__(self, cfgFilePath = ''):
        self.state = {}
        self.state['word_emb_file']    = ''
        self.state['word_emb_size']    = 100
        self.state['word_vocab_file']  = '../file/vocab.word.txt'
        
        self.state['kb_emb_file']      = '/disk1/weilin2/KBP/EDL/transE/model.lr0001.dim50.bSize32.gama1.txt'
        self.state['kb_emb_size']      = 50
        self.state['kb_vocab_file']    = '../file/vocab.kb.txt'
        
        self.state['kbp_type_vocab_file'] = '../file/vocab.kbp.type.txt'
        self.state['kbp_type_emb_size']   = 3
        self.state['fb_type_vocab_file']  = '../file/vocab.fb.type.txt'
        self.state['fb_type_emb_size']    = 3
        
        self.state['hid1_size'] = 500
        self.state['hid2_size'] = 300
        self.state['init_lr'] = 0.01
        self.state['weight_decay'] = 0.
        self.state['batch_size'] = 1
        self.state['train'] = '../file/train.f.txt'
        self.state['test'] = '../file/test.f.txt'
        self.state['trainAttFile'] = ''
        self.state['testAttFile'] = ''
        self.state['node_hot'] = '../file/NodeHot.txt'
        self.state['object_muti_name'] = '../file/object_muti_name.txt'
        self.state['valid'] = ''
        self.state['epoc'] = 50
        self.state['declr_start'] = 10
        self.state['outdir'] = ''
        self.state['static_emb'] = False
        self.state['pre_model'] = None
        self.state['model'] = '../file/RankModel'
        self.state['test_model'] = ''
        self.state['ran_seed'] = 1
        self.state['cache_size'] = 20000
        self.state['debug_model'] = 1
        self.state['string_match_vocab_size'] = 6
        self.state['string_match_emb_size']   = 10
        self.state['word_emb_file']           = '../file/embedding.txt'
        
        
        self.loadFromFile(cfgFilePath)
        self.printInfo()
    
    def loadFromFile(self, cfgFilePath):
        if(os.path.exists(cfgFilePath)):
            print 'load parameters from file: ', cfgFilePath
            lines = open(cfgFilePath, 'rb').readlines()
            for line in lines:
                if(line[0:1] == '#'):
                    continue
                else:
                    info = line.strip().split('=')
                    if(len(info) == 2):
                        key = info[0].strip()
                        val = info[1].strip()
                        if(val.isdigit()):
                            if(key == "init_lr" or key == "weight_decay"):
                                val = float(val)
                            else:
                                val = int(val)
                        if(self.state.has_key(key)):
                            self.state[key] = val
        else:
            print 'open cfgFile error, use default parameters. cfgFilePath: ', cfgFilePath
      
    def printInfo(self):
        print 'cfg information:'
        for k,v in self.state.items():
            print k, ' = ', v