#!/usr/bin/env python
import numpy as np
from pylab import imread, gray
from matplotlib.pyplot import imshow, figure, show
import random
from messagepassing import Node, Variable, Factor, bcolors

def prepare():
    # Load the image and binarize
    im = np.mean(imread('dalmatian1.png'), axis=2) > 0.5
    #imshow(im)
    #gray()

    # Add some noise
    noise = np.random.rand(*im.shape) > 0.9
    noise_im = np.logical_xor(noise, im)
    #figure()
    #imshow(noise_im)

    test_im = np.zeros((10,10))
    #test_im[5:8, 3:8] = 1.0
    #test_im[5,5] = 1.0
    #figure()
    #imshow(test_im)

    # Add some noise
    noise = np.random.rand(*test_im.shape) > 0.9
    noise_test_im = np.logical_xor(noise, test_im)
    #figure()
    #imshow(noise_test_im)
    #show()
    return noise_im[0:200,0:220]

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
    vertical_factors = []
    horizontal_factors = []
    depth_factors = []
    print "vertical factors (latent variables)"
    for x in range(im.shape[1]):
        for y in range(im.shape[0]-1):
            fname = str(x)+":"+str(y)+"_"+str(x)+":"+str(y+1)
            neighbours = [lg[y][x],lg[y+1][x]]
            f = np.array([[0.9,0.1],[0.1,0.9]])
            #f = np.array(np.ones((2,2))/4) # uniform distribution
            fa = Factor(fname,f,neighbours)
            vertical_factors.append(fa)

    print "horizontal factors (latent variables).."
    for x in range(im.shape[1]-1):
        for y in range(im.shape[0]):
            fname = str(x)+":"+str(y)+"_"+str(x+1)+":"+str(y)
            neighbours = [lg[y][x],lg[y][x+1]]
            f = np.array([[0.9,0.1],[0.1,0.9]])
            #f = np.array(np.ones((2,2))/4) # uniform distribution
            fa = Factor(fname,f,neighbours)
            horizontal_factors.append(fa)

    print '"depth" factors (observed/latent variables)..'
    for x in range(im.shape[1]):
        for y in range(im.shape[0]):
            fname = "o_"+str(x)+":"+str(y)+"_"+str(x)+":"+str(y)
            neighbours = [lg[y][x],og[y][x]]
            f = np.array([[0.9,0.1],[0.1,0.9]])
            #f = np.array(np.ones((2,2))/4) # uniform distribution
            fa = Factor(fname,f,neighbours)
            depth_factors.append(fa)
    print "done creating factors."
    return vertical_factors,horizontal_factors,depth_factors

def create_image(lg):
    y_max = len(lg)
    x_max = len(lg[0])
    ret = np.zeros((y_max,x_max))
    for y in range(y_max):
        for x in range(x_max):
            node = lg[y][x]
            value = node.argmax('ms')[0]
            ret[y,x] = value

    return ret

def main_dalmatian():
    noise_im = prepare()
    #noise_im = np.zeros((10,10))
    #noise_im[5:9,5:9] = 1

    lg,og = create_vargrids(noise_im)
    fa1,fa2,fa3 = create_factors(noise_im,lg,og)
    factors = fa1+fa2+fa3
    all_nodes = [v for sublist in lg + og for v in sublist]+factors
    for node in all_nodes:
        r = random.random()
        msg_init = np.array([r, 1-r])
        node.initialize_messages(msg_init)
        node.set_pending_except()

    num_iterations = 7
    for iteration in range(num_iterations):
        print "iteration",iteration,"..."
        shuffled_nodes = all_nodes[:]
        random.shuffle(shuffled_nodes)
        for node in shuffled_nodes:
            #print "node",node.name
            node.send_pending('ms')

    recovered = create_image(lg)
    figure()
    gray()
    imshow(noise_im)
    figure()
    gray()
    imshow(recovered)
    show()
    print "done."

if __name__ == "__main__":
    main_dalmatian()
