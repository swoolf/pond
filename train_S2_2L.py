import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
#import driving_data
import S2_data
import model

LOGDIR = './saveS2_2L'

sess = tf.InteractiveSession()

L2NormConst = 0.001
YNormConst = 100
train_vars = tf.trainable_variables()

loss = YNormConst*tf.reduce_mean(tf.square(tf.sub(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
vloss = YNormConst*tf.reduce_mean(tf.square(tf.sub(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
error = 1-tf.reduce_mean( tf.divide(tf.abs(tf.sub(model.y_, model.y)), model.y) )
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
sess.run(tf.initialize_all_variables())

# create a summary to monitor cost tensor
#tf.scalar_summary("loss", loss)
#tf.scalar_summary("vloss", vloss)
#merge all summaries into a single op
#merged_summary_op = tf.merge_all_summaries()

loss_summary_op=tf.summary.merge([tf.scalar_summary("loss", loss)])
vloss_summary_op=tf.summary.merge([tf.scalar_summary("vloss", vloss), tf.scalar_summary("v_error", error)])

saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)

# op to write logs to Tensorboard
logs_path = './logsS2_2L'
summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

epochs = 200
batch_size = 50

# train over the dataset about 30 times
for epoch in range(epochs):
  for i in range(int(S2_data.num_images/batch_size)):
    xs, ys = S2_data.LoadTrainBatch(batch_size)
    train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8})
    if i % 10 == 0:
      vxs, vys = S2_data.LoadValBatch(batch_size)
      vloss_value = vloss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, vloss_value))
      #summary = tf.Summary(value=[ tf.Summary.Value(tag="valoss", simple_value=vloss_value),])
      #summary_writer.add_summary(summary, epoch * S2_data.num_images/batch_size + i)      
      summary = vloss_summary_op.eval(feed_dict={model.x:vxs, model.y_: vys, model.keep_prob: 1.0})
      summary_writer.add_summary(summary, epoch * S2_data.num_images/batch_size + i)


    # write logs at every iteration
    summary = loss_summary_op.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
    summary_writer.add_summary(summary, epoch * S2_data.num_images/batch_size + i)

    if i % batch_size == 0:
      if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
      checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
      filename = saver.save(sess, checkpoint_path)
  print("Model saved in file: %s" % filename)

print("Run the command line:\n" \
          "--> tensorboard --logdir=./logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
