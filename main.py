import os
# suppress excess messaging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
import helper
import warnings
import time
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# hyperparameters
STD_DEV = 0.01
L2_REG = 0.01
KEEP_PROB = 0.5
LEARNING_RATE = 1e-4
EPOCHS = 20
REG_CONSTANT = 0.00001

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    # scaling (per advice of pierluigi.ferrari)
    layer4_scaled = tf.multiply(vgg_layer4_out, 0.01, name='layer4_scaled')
    layer3_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='layer3_scaled')

    # 1x1 convolutions
    layer7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes,
            kernel_size=1, strides=(1,1), padding='SAME',
            kernel_initializer=tf.random_normal_initializer(stddev=STD_DEV),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG), name='layer7_1x1')
    layer4_1x1 = tf.layers.conv2d(layer4_scaled, num_classes,
            kernel_size=1, strides=(1,1), padding='SAME',
            kernel_initializer=tf.random_normal_initializer(stddev=STD_DEV),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG), name='layer4_1x1')
    layer3_1x1 = tf.layers.conv2d(layer3_scaled, num_classes,
            kernel_size=1, strides=(1,1), padding='SAME',
            kernel_initializer=tf.random_normal_initializer(stddev=STD_DEV),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG), name='layer3_1x1')

    # upsample
    layer7_2x = tf.layers.conv2d_transpose(layer7_1x1, num_classes,
            kernel_size=4, strides=(2,2), padding='SAME',
            kernel_initializer=tf.random_normal_initializer(stddev=STD_DEV),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG), name='layer7_2x')

    # skip connection
    layer4_skip = tf.add(layer7_2x, layer4_1x1)

    # upsample
    layer4_2x = tf.layers.conv2d_transpose(layer4_skip, num_classes,
            kernel_size=4, strides=(2,2), padding='SAME',
            kernel_initializer=tf.random_normal_initializer(stddev=STD_DEV),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG), name='layer4_2x')
    
    # skip connection
    layer3_skip = tf.add(layer4_2x, layer3_1x1)

    # upsample
    layer_output = tf.layers.conv2d_transpose(layer3_skip, num_classes,
            kernel_size=16, strides=(8,8), padding='SAME',
            kernel_initializer=tf.random_normal_initializer(stddev=STD_DEV),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG), name='layer_output')

    return layer_output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # loss functions
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    combined_loss = cross_entropy_loss + REG_CONSTANT * sum(reg_losses)

    # training operation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(combined_loss)

    return logits, train_op, combined_loss
tests.test_optimize(optimize)


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
    # TODO: Implement function

    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    print("EPOCH\tloss\ttime (s)")

    for epoch in range(epochs):
        for image, label in get_batches_fn(batch_size):
            # Training
            _, loss = sess.run([train_op, cross_entropy_loss],
                        feed_dict={input_image: image, correct_label: label,
                        keep_prob: KEEP_PROB, learning_rate: LEARNING_RATE})
        end_time = time.time()
        print("%3d %9.4f %8.2f" % (epoch + 1, loss, end_time - start_time))
        start_time = end_time

tests.test_train_nn(train_nn)

def run_tests():
    print(">>>> tests.test_load_vgg(load_vgg, tf)")
    tests.test_load_vgg(load_vgg, tf)

    print(">>>> tests.test_layers(layers)")
    tests.test_layers(layers)

    print(">>>> tests.test_optimize(optimize)")
    tests.test_optimize(optimize)

    print(">>>> tests.test_train_nn(train_nn)")
    tests.test_train_nn(train_nn)

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    batch_size = 8

    # Create a TensorFlow configuration object. This will be
    # passed as an argument to the session.
    config = tf.ConfigProto()

    # JIT level, this can be set to ON_1 or ON_2
    jit_level = tf.OptimizerOptions.ON_1
    config.graph_options.optimizer_options.global_jit_level = jit_level

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')

    # Create function to get batches
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

    # OPTIONAL: Augment Images for better results
    # https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

    with tf.Session(config=config) as sess:

        correct_label = tf.placeholder(tf.int32)
        learning_rate = tf.placeholder(tf.float32)

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, EPOCHS, batch_size, get_batches_fn, train_op, loss, input_image,
             correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        start_time = time.time()
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        elapsed_time = time.time() - start_time
        print("Elapsed time for classification test: %6.2f s" % (elapsed_time))

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    #run_tests()
    run()
