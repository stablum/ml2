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
    print "creating variable grids.."
    lg = [] # grid of latents
    og = [] # grid of observed
    for y in range(im.shape[0]):
        lg.append([])
        og.append([])
        for x in range(im.shape[1]):
            lname = "l_"+str(x)+":"+str(y)
            l = Variable(lname,2)
            l.set_latent()
            lg[y].append(l)

            oname = "o_"+str(x)+":"+str(y)
            o = Variable(oname,2)
            o.set_observed(im[y,x])
            og[y].append(o)
    print "done creating variable grids"
    return lg, og

def create_factors(im,lg,og):
    print "creating factors.."
    ret = []
    print "vertical factors (latent variables)"
    for x in range(im.shape[1]):
        for y in range(im.shape[0]-1):
            fname = str(x)+":"+str(y)+"_"+str(x)+":"+str(y+1)
            neighbours = [lg[y][x],lg[y+1][x]]
            f = np.array(np.ones((2,2))/4) # uniform distribution
            fa = Factor(fname,f,neighbours)
            ret.append(fa)

    print "horizontal factors (latent variables).."
    for x in range(im.shape[1]-1):
        for y in range(im.shape[0]):
            fname = str(x)+":"+str(y)+"_"+str(x+1)+":"+str(y)
            neighbours = [lg[y][x],lg[y][x+1]]
            f = np.array(np.ones((2,2))/4) # uniform distribution
            fa = Factor(fname,f,neighbours)
            ret.append(fa)
    return ret

    print '"depth" factors (observed/latent variables)..'
    for x in range(im.shape[1]):
        for y in range(im.shape[0]):
            fname = "o_"+str(x)+":"+str(y)+"_"+str(x)+":"+str(y)
            neighbours = [lg[y][x],og[y][x]]
            f = numpy.array(np.ones((2,2))/4) # uniform distribution
            fa = Factor(fname,f,neighbours)
            ret.append(fa)
    print "done creating factors."
    return ret

lg,og = create_vargrids(noise_im)
factors = create_factors(noise_im,lg,og)
all_nodes = [v for sublist in lg + og for v in sublist]+factors
for node in all_nodes:
    node.initialize_messages(0)
    node.set_pending_except()

num_iterations = 10
for iteration in range(num_iterations):
    print "iteration",iteration,"..."
    for node in all_nodes:
        node.send_pending('ms')
print "size all nodes",len(all_nodes)
print "done."

