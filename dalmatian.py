#!/usr/bin/env python
import numpy as np
from pylab import imread, gray
from matplotlib.pyplot import imshow, figure, show

from messagepassing import Node, Variable, Factor, bcolors

# Load the image and binarize
im = np.mean(imread('dalmatian1.png'), axis=2) > 0.5
imshow(im)
gray()

# Add some noise
noise = np.random.rand(*im.shape) > 0.9
noise_im = np.logical_xor(noise, im)
figure()
imshow(noise_im)

test_im = np.zeros((10,10))
#test_im[5:8, 3:8] = 1.0
#test_im[5,5] = 1.0
figure()
imshow(test_im)

# Add some noise
noise = np.random.rand(*test_im.shape) > 0.9
noise_test_im = np.logical_xor(noise, test_im)
figure()
imshow(noise_test_im)

#show()

def create_vargrids(im):
    lg = [] # grid of latents
    og = [] # grid of observed
    for x in range(im.shape[1]):
        lg.append([])
        og.append([])
        for y in range(im.shape[0]):
            lname = str(x)+":"+str(y)
            l = Variable(lname,2)
            l.set_latent()
            lg[x].append(l)

            oname = "o_"+str(x)+":"+str(y)
            o = Variable(oname,2)
            o.set_observed(im[y,x])
            og[x].append(o)

    return lg, og

def create_factors(im,lg,og):
    ret = []
    # vertical factors (latent variables)
    for x in range(im.shape[1]):
        for y in range(im.shape[0]-1):
            fname = str(x)+":"+str(y)+"_"+str(x)+":"+str(y+1)
            neighbours = [lg[y][x],lg[y+1][x]]
            f = np.array(np.ones((2,2))/4) # uniform distribution
            fa = Factor(fname,f,neighbours)
            ret.append(fa)

    # horizontal factors (latent variables)
    for x in range(im.shape[1]-1):
        for y in range(im.shape[0]):
            fname = str(x)+":"+str(y)+"_"+str(x+1)+":"+str(y)
            neighbours = [lg[y][x],lg[y][x+1]]
            f = np.array(np.ones((2,2))/4) # uniform distribution
            fa = Factor(fname,f,neighbours)
            ret.append(fa)
    return ret

    # "depth" factors (observed/latent variables)
    for x in range(im.shape[1]):
        for y in range(im.shape[0]):
            fname = "o_"+str(x)+":"+str(y)+"_"+str(x)+":"+str(y)
            neighbours = [lg[y][x],og[y][x]]
            f = numpy.array(np.ones((2,2))/4) # uniform distribution
            fa = Factor(fname,f,neighbours)
            ret.append(fa)
    return ret

lg,og = create_vargrids(test_im)
factors = create_factors(test_im,lg,og)
print "done."
