import tensorflow as tf
import scipy.misc
import model
import random 

import matplotlib as mp
import matplotlib.pyplot as plt

def getActivations(layer,image):
    units = sess.run(layer,feed_dict={model.x: [image], model.keep_prob: 1.0})
    plotNNFilter(units)
    
def plotNNFilter(units):
    filters = units.shape[3]
#    plt.figure(1, figsize=(20,20))
#    n_columns = 6
#    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
       	plt.imsave('activations/3filter'+str(i)+'.png', units[0,:,:,i], cmap="gray")
#        plt.subplot(n_rows, n_columns, i+1)
#        plt.title('Filter ' + str(i))
#        plt.savefig('filter'+str(i)+'.png', bbox_inches='tight')
#        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
    
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

xs = []
ys = []

with open("/cluster/shared/swoolf02/trimData/data.txt") as f:
    for line in f:
        xs.append("/cluster/shared/swoolf02/trimData/" + line.split()[0])
        ys.append(float(line.split()[1])-366.49)
        
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

pic  = "/cluster/shared/swoolf02/trimData/170310_2343.jpg"

full_image = scipy.misc.imread(pic, mode="RGB")
image = scipy.misc.imresize(full_image, [100, 200]) / 255.0
plt.imsave('activations/hmmm2.png',image)
height = ys[0]
#print(height)
getActivations(model.h_conv3,image)
