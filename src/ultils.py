import random
import sys
import traceback
import numpy
import theano
import theano.tensor as TT
import string

def print_mat(x):
    if x.ndim == 2:
        for i in xrange(x.shape[0]):
            print '%d,' % x.shape[0]
            for j in xrange(x.shape[1]):
                print '%f  %d,' % (x[i, j], x.shape[1]),
            print ']'
    elif x.ndim == 1:
        for j in xrange(x.shape[0]):
            print '%f,' % x[j]
        print ']'
    elif x.ndim == 3:
        for i in xrange(x.shape[0]):
            for j in xrange(x.shape[1]):
                for k in xrange(x.shape[2]):
                    print '%f,' %x[i, j, k],
                print ']'
            print ']'
    elif x.ndim == 4:
        for i in xrange(x.shape[0]):
            for j in xrange(x.shape[1]):
                for k in xrange(x.shape[2]):
                    for l in xrange(x.shape[3]):
                        print '%f,' %x[i, j, k, l]
                    print ']'
                print ']'
            print ']'
    else:
        print x.__str__()
    return x

def print_hook(op, cnda):
    print op.message, '=',
    print_mat(cnda)
def print_hook_shape(op, cnda):
    print op.message, '=',
    print cnda.shape
def dbg_hook(msg, x):
    return theano.printing.Print(message=msg, global_fn=print_hook)(x)
def dbg_hook_shape(msg, x):
    return theano.printing.Print(message=msg, global_fn=print_hook_shape)(x)



def id_generator(size = 5, chars = string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for i in xrange(size))

def sample_weights(sizeX, sizeY, sparsity, scale, rng):
    sizeX = int(sizeX)
    sizeY = int(sizeY)
    sparsity = numpy.minimum(sizeY, sparsity)
    values = numpy.zeros((sizeX, sizeY), dtype = theano.config.floatX)
    for dx in xrange(sizeX):
        perm = rng.permutation(sizeY)
        new_vals = rng.uniform(low = -scale, high = scale, size = (sparsity,))
        vals_norm = numpy.sqrt((new_vals**2).sum())
        new_vals = scale * new_vals / vals_norm
        values[dx, perm[:sparsity]] = new_vals
    _, v, _ = numpy.linalg.svd(values)
    values = scale * values / v[0]
    return values.astype(theano.config.floatX)

def sample_weights_classic(sizeX, sizeY, sparsity, scale, rng):
    sizeX = int(sizeX)
    sizeY = int(sizeY)
    if sparsity < 0:
        sparsity = sizeY
    else:
        sparsity = numpy.minimun(sizeY, sparsity)
    sparsity = numpy.minimun(sizeY, sparsity)
    values = numpy.zeros((sizeX, sizeY), dtype = theano.config.floatX)
    for dx in xrange(sizeX):
        perm = rgn.permutation(sizeY)
        new_vals = rng.normal(loc=0, scale=scale, size=(sparsity,))
        values[dx, perm[:sparsity]] = new_vals
    return values.astype(theano.config.floatX)

def sample_weights_sqrt(sizeX, sizeY, sparsity, scale, rng):
    sizeX = int(sizeX)
    sizeY = int(sizeY)
    bnd= numpy.sqrt(6./(sizeX+sizeY))
    values= numpy.asarray(rng.uniform(low=-bnd,high=bnd,size=(sizeX,sizeY)),dtype = 'float32')
    return values

    
def sample_weights_orth(sizeX, sizeY, sparsity, scale, rng):
    sizeX = int(sizeX)
    sizeY = int(sizeY)

    assert sizeX == sizeY, 'for orthogonal init, sizeX == sizeY'
    if sparsity < 0:
        sparsity = sizeY
    else:
        sparsity = numpy.minimum(sizeY, sparsity)
    values = numpy.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in xrange(sizeX):
        perm = rng.permutation(sizeY)
        new_vals = rng.normal(loc=0, scale=scale, size=(sparsity,))
        values[dx, perm[:sparsity]] = new_vals
    u,s,v = numpy.linalg.svd(values)
    values = u * scale
    return values.astype(theano.config.floatX)

def init_bias(size, scale, rng):
    return numpy.zeros((size,), dtype=theano.config.floatX)*float(scale)
    #return numpy.asarray(scale * rng.standard_normal(size = (size)), dtype = theano.config.floatX)

ReLU = lambda x: TT.maximum(0.0,x)
Sigmoid = lambda x: TT.nnet.sigmoid(x)
Tanh = lambda x: TT.tanh(x)
Ident = lambda x:x
