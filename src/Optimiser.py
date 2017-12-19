from collections import OrderedDict
import cPickle as pkl
import sys
import time
import numpy
import theano
import theano.tensor as tensor

#class Normal(object):
#    def __init__(self, tparms):
        
class Sgd(object):
    def __init__(self, tparms, momentum = 0.9):
        self.momentum = 0.9
        self.parm_dict = {}
        for tp in tparms:
            gshared = theano.shared(tp.get_value() * 0, name = '%s_grad' % tp.name)
            gpre = theano.shared(tp.get_value() * 0, name = '%s_pre' % tp.name)
            self.parm_dict[tp.name] = (tp, gshared, gpre)
    def Optimiser(self, tparms, grads, inputs, cost):
        gshared = [self.parm_dict[tp.name][1] for tp in tparms]
        gpre = [self.parm_dict[tp.name][2] for tp in tparms]
        lr = tensor.scalar(name = 'lr')
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        gsup = [(gs, g) for gs, g in zip(gshared, grads)]
        f_grad_shared = theano.function(inputs, cost, updates = gsup, allow_input_downcast = True, name = 'sgd_f_grad_shared',on_unused_input = 'ignore')
        up_grad = [gp * self.momentum + lr * gs for gs, gp in zip(gshared, gpre)]
        preup = [(gp, ud) for gp, ud in zip(gpre, up_grad)]
        pup = [(p, p - ud) for p, ud in zip(tparms, up_grad)]
        f_update = theano.function([lr], [], updates = pup + preup, allow_input_downcast = True, name = 'sgd_f_update',on_unused_input = 'ignore')
        
        return f_grad_shared, f_update
    
class AdaDelta(object):
    
    def __init__(self, tparms):
        self.parm_dict = {}
        for p in tparms:
            zg = theano.shared(p.get_value() * 0, name = '%s_grad' % p.name)
            rup2 = theano.shared(p.get_value() * 0, name = '%s_rup2' % p.name)
            rgrad2 = theano.shared(p.get_value() * 0, name = '%s_rgrad2' % p.name)
            self.parm_dict[p.name] = (p, zg, rup2, rgrad2)
    
    def Optimiser(self, tparms, grads, inputs, cost):
        lr = tensor.scalar(name = 'lr')
        if not isinstance(inputs, (list, tuple)):
            inputs=[inputs]
        zipped_grads =[self.parm_dict[p.name][1] for p in tparms]
        running_up2 =[self.parm_dict[p.name][2] for p in tparms]
        running_grads2 =[self.parm_dict[p.name][3] for p in tparms]
        
        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) 
                 for rg2, g in zip(running_grads2, grads)]
        
        f_grad_shared = theano.function(inputs, cost, updates=zgup + rg2up, allow_input_downcast = True,
                                        name = 'adadelta_f_grad_shared')
        updir = [-tensor.sqrt(ru2 + 1e-7) / tensor.sqrt(rg2 + 1e-7) * zg
                 for zg, ru2, rg2 in zip(zipped_grads,
                                         running_up2,
                                         running_grads2)]
        ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
                 for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + ud) for p, ud in zip(tparms, updir)]
        f_update = theano.function([lr], [], updates = ru2up + param_up, 
                                   on_unused_input = 'ignore', allow_input_downcast = True, name = 'adadelta_f_update')
        
        return f_grad_shared, f_update
        #allow_input_downcast = True,
