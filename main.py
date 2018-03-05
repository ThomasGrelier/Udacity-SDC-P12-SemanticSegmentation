import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

#%%
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess,[vgg_tag],vgg_path)
    graph = tf.get_default_graph()
    
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out
#tests.test_load_vgg(load_vgg, tf)

#%%
def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    if 0: # print tensor shapes
        layer3_tensor_shape = vgg_layer3_out.get_shape().as_list()
        layer4_tensor_shape = vgg_layer4_out.get_shape().as_list()
        layer7_tensor_shape = vgg_layer7_out.get_shape().as_list()
        print('Shape :{}'.format(layer3_tensor_shape)) 
        print('Shape :{}'.format(layer4_tensor_shape))
        print('Shape :{}'.format(layer7_tensor_shape))
    
    # Parameters
    reg_param = 1e-3
    std_kern_weight = 0.01
    scale_factor_3 = 0.0001
    scale_factor_4 = 0.01
    

    # Layer 7 1x1 conv
    vgg_layer7_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1, 1),\
                    padding='same',
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_param),\
                    kernel_initializer=tf.truncated_normal_initializer(stddev=std_kern_weight))
    # Layer upsampling (x2)
    vgg_layer7_out = tf.layers.conv2d_transpose(vgg_layer7_out, num_classes, 4,\
                    strides=(2, 2), padding='same',\
                    kernel_initializer=tf.truncated_normal_initializer(stddev=std_kern_weight), \
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_param))
    # Scale layer 4
    vgg_layer4_out = tf.scalar_mul(scale_factor_4, vgg_layer4_out)
    # Layer 4 1x1 conv
    vgg_layer4_out = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1, 1), \
                    padding='same', \
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_param), \
                    kernel_initializer=tf.truncated_normal_initializer(stddev=std_kern_weight))
    # Add modified layer 4 and 7
    layer = tf.add(vgg_layer7_out, vgg_layer4_out)
    # Layer upsampling (x2)
    layer = tf.layers.conv2d_transpose(layer, num_classes, 4, strides=(2, 2),\
                    padding='same', \
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_param),\
                    kernel_initializer=tf.truncated_normal_initializer(stddev=std_kern_weight))
    # Scale layer 3
    vgg_layer3_out = tf.scalar_mul(scale_factor_3, vgg_layer3_out)
    # Layer 3 1x1 conv
    vgg_layer3_out = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1, 1),\
                    padding='same', \
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_param),\
                    kernel_initializer=tf.truncated_normal_initializer(stddev=std_kern_weight))
    # Add modified layer 3, 4 and 7
    layer = tf.add(layer, vgg_layer3_out)
    # Layer upsampling (x8)
    layer = tf.layers.conv2d_transpose(layer, num_classes, 16, strides=(8, 8), \
                    padding='same', \
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_param),\
                    kernel_initializer=tf.truncated_normal_initializer(stddev=std_kern_weight))
    return layer
#tests.test_layers(layers)

    
#%%
def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    reg_ws = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    for w in reg_ws:
        shp = w.get_shape().as_list()
        print("- {} shape:{} size:{}".format(w.name, shp, np.prod(shp)))
    print("")
    
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=correct_label)) \
                        + tf.reduce_sum(reg_ws)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
#tests.test_optimize(optimize)

#%%
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    
    print("Training...")
    sess.run(tf.global_variables_initializer())
    keep_prob_ = 0.5
    learning_rate_ = 0.0001
    for i in range(epochs):
        for batch_x, batch_y in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: batch_x, correct_label: batch_y, \
                               learning_rate: learning_rate_, keep_prob: keep_prob_})  
        print("Epoch: {}, Training loss = {:.3f}".format(i+1,loss))    
    
#tests.test_train_nn(train_nn)

#%%
def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    epochs = 40
    batch_size = 8
    
    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # Build NN using load_vgg, layers, and optimize function
        learning_rate = tf.placeholder(tf.float32)
        keep_prob = tf.placeholder(tf.float32)
        correct_label = tf.placeholder(tf.float32, (None, None, None, 2))
        
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

    
if __name__ == '__main__':
    run()
