import tensorflow as tf
import scipy.misc
import model
import random 

sess = tf.InteractiveSession()
saver = tf.train.Saver()
#saver.restore(sess, "saveS2_2L/model.ckpt")
#saver.restore(sess, "saveS1_2L/model.ckpt")
saver.restore(sess, "saveS2_fl/model.ckpt")
#saver.restore(sess, "saveS2_fl_forward/model.ckpt")

xs = []
ys = []

#read data.txt
#with open("/cluster/shared/swoolf02/o_data/data_test.txt") as f:
#with open("/cluster/shared/swoolf02/val2Data/data.txt") as f:
with open("/cluster/shared/swoolf02/o_data/data_test_fl.txt") as f:
#with open("/cluster/shared/swoolf02/o_data/data_test_fl_forward.txt") as f:
    for line in f:
        xs.append("/cluster/shared/swoolf02/o_data/" + line.split()[0])
#	xs.append("/cluster/shared/swoolf02/val2Data/" + line.split()[0])
        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output
        #ys.append(float(line.split()[1]) * scipy.pi / 180)
        #ys.append((float(line.split()[1])-366.49)*12.0)
#        ys.append((float(line.split()[1])-366.49)/6.23)
	ys.append((float(line.split()[1])-21)/95)
	# account for sealevel
        
#shuffle list of images
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

fracTest = 1

test_xs = xs[:int(len(xs) * fracTest)]
test_ys = ys[:int(len(xs) * fracTest)]
num_images = len(test_xs)

output=[]
for i in range(0,num_images):
    full_image = scipy.misc.imread(test_xs[i], mode="RGB")
    image = scipy.misc.imresize(full_image, [100, 200]) / 255.0
    height = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0]
    output.append([test_xs[i], test_ys[i], height])
    #print [test_xs[i], test_ys[i], height]
    print test_xs[i], height, test_ys[i]
