import logging
import numpy as np
import tensorflow as tf

logger_handler = logging.StreamHandler()
logger_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logging.root.handlers = [logger_handler]
logging.root.setLevel(logging.DEBUG)


class Dataset:
    def __init__(self, x, y, batch_size, shuffle=False):
        assert x.shape[0] == y.shape[0], 'got different numbers of data and labels'
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        n, batch_size = self.x.shape[0], self.batch_size
        indexes = np.arange(n)
        if self.shuffle:
            np.random.shuffle(indexes)
        return iter(
            (self.x[indexes[i:i + batch_size]], self.y[indexes[i:i + batch_size]])
            for i in range(0, n, batch_size)
        )


def train(
        device,
        train_dataset,
        val_dataset,
        forward_model_graph,
        optimizer,
        outer_loop_iterations=15,
        print_every=100,
):
    # def learn(scores, y, params, learning_rate):
    #     # First compute the loss; the first line gives losses for each example in
    #     # the minibatch, and the second averages the losses acros the batch
    #     losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
    #     loss = tf.reduce_mean(losses)
    #
    #     # Compute the gradient of the loss with respect to each parameter of the
    #     # network. This is a very magical function call: TensorFlow internally
    #     # traverses the computational graph starting at loss backward to each element
    #     # of params, and uses backpropagation to figure out how to compute gradients;
    #     # it then adds new operations to the computational graph which compute the
    #     # requested gradients, and returns a list of TensorFlow Tensors that will
    #     # contain the requested gradients when evaluated.
    #     param_gradients = tf.gradients(loss, params)
    #
    #     # Make a gradient descent step on all of the model parameters.
    #     new_params = []
    #     for param, param_gradient in zip(params, param_gradients):
    #         new_param = tf.assign_sub(param, learning_rate * param_gradient)
    #         new_params.append(new_param)
    #
    #     # Insert a control dependency so that evaluting the loss causes a weight
    #     # update to happen; see the discussion above.
    #     with tf.control_dependencies(new_params):
    #         return tf.identity(loss)

    def calculate_accuracy():
        num_correct, num_samples = 0, 0
        for x_val_batch, y_val_batch in val_dataset:
            calculated_scores = session.run(scores, feed_dict={
                x: x_val_batch,
                is_training: False,
            })
            y_pred = calculated_scores.argmax(axis=1)
            num_samples += x_val_batch.shape[0]
            num_correct += (y_pred == y_val_batch).sum()
        accuracy = float(num_correct) / num_samples
        return accuracy

    # First clear the default graph
    tf.reset_default_graph()
    is_training = tf.placeholder(tf.bool, name='is_training')

    # Set up the computational graph for performing forward and backward passes,
    # and weight updates.
    with tf.device(device):
        # Set up placeholders for the data and labels
        x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        y = tf.placeholder(tf.int32, [None])

        scores, l2_loss = forward_model_graph(x, is_training)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
        loss = tf.reduce_mean(losses) + 0.001 * l2_loss
        train_step = optimizer.minimize(loss)

        # train_step = learn(scores, y, params, learning_rate)

    # Now we actually run the graph many times using the training data
    with tf.Session() as session:
        # Initialize variables that will live in the graph
        session.run(tf.global_variables_initializer())

        for outer_loop_iteration in range(outer_loop_iterations):
            logging.debug(f"outer loop iteration {outer_loop_iteration}")

            for batch_index, (x_train_batch, y_train_batch) in enumerate(train_dataset):
                # Run the graph on a batch of training data; recall that asking
                # TensorFlow to evaluate loss will cause an SGD step to happen.
                session.run(train_step, feed_dict={
                    x: x_train_batch,
                    y: y_train_batch,
                    is_training: True
                })

                # Periodically print the loss and check accuracy on the val set
                if batch_index % print_every != 0:
                    continue

                calculated_loss = session.run(loss, feed_dict={
                    x: x_train_batch,
                    y: y_train_batch,
                    is_training: True
                })

                logging.debug(f"accuracy: {calculate_accuracy()} \t| loss: {calculated_loss}")

        return calculate_accuracy()


###################################################################################################


def custom_cnn(input, is_training):
    def kaiming_normal(shape):
        if len(shape) == 2:
            fan_in, fan_out = shape[0], shape[1]
        elif len(shape) == 4:
            fan_in, fan_out = np.prod(shape[:3]), shape[3]
        else:
            raise AssertionError
        return tf.random_normal(shape) * np.sqrt(2.0 / fan_in)

    def conv2d(input, fh, fw, pf, f):
        weights = tf.Variable(kaiming_normal((fh, fw, pf, f)))
        bias = tf.Variable(tf.zeros(f))
        # return tf.nn.local_response_normalization(
        #     tf.nn.relu(
        #         tf.nn.bias_add(
        #             tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding="SAME"),
        #             bias,
        #         ),
        #     )
        # )
        return tf.nn.relu(
            tf.nn.bias_add(
                tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding="SAME"),
                bias,
            ),
        ), tf.nn.l2_loss(weights)

    def max_pool2(input):
        return tf.nn.max_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

    def fc(input, rows, cols):
        weights = tf.Variable(kaiming_normal((rows, cols)))
        bias = tf.Variable(tf.zeros(cols))
        return tf.matmul(input, weights) + bias, tf.nn.l2_loss(weights)

    # Based on VGGNet model

    N = tf.shape(input)[0]

    conv1, l2_loss1 = conv2d(input, 3, 3, 3, 64)
    conv2, l2_loss2 = conv2d(conv1, 3, 3, 64, 64)
    conv3 = max_pool2(conv2)

    conv4, l2_loss4 = conv2d(conv3, 3, 3, 64, 128)
    conv5, l2_loss5 = conv2d(conv4, 3, 3, 128, 128)
    conv6 = max_pool2(conv5)

    conv7, l2_loss7 = conv2d(conv6, 3, 3, 128, 256)
    conv8, l2_loss8 = conv2d(conv7, 3, 3, 256, 256)
    conv9, l2_loss9 = conv2d(conv8, 3, 3, 256, 256)
    conv10 = max_pool2(conv9)

    conv11, l2_loss11 = conv2d(conv10, 3, 3, 256, 512)
    conv12, l2_loss12 = conv2d(conv11, 3, 3, 512, 512)
    conv13, l2_loss13 = conv2d(conv12, 3, 3, 512, 512)
    conv14 = max_pool2(conv13)

    flat = tf.reshape(conv14, (N, -1))

    fc1, l2_loss_fc1 = fc(flat, 2 * 2 * 512, 4096)
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob=0.5)

    fc2, l2_loss_fc2 = fc(fc1, 4096, 4096)
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob=0.5)

    fc3, l2_loss_fc3 = fc(fc2, 4096, 10)

    scores = fc3

    l2_loss = l2_loss1 + \
              l2_loss2 + \
              l2_loss4 + \
              l2_loss5 + \
              l2_loss7 + \
              l2_loss8 + \
              l2_loss9 + \
              l2_loss11 + \
              l2_loss12 + \
              l2_loss13 + \
              l2_loss_fc1 + \
              l2_loss_fc2 + \
              l2_loss_fc3

    return scores, l2_loss


###################################################################################################

def load_cifar10(num_training=49000, num_validation=1000, num_test=10000):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).flatten()
    x_test = np.asarray(x_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).flatten()

    # Subsample the data

    val_mask = range(num_training, num_training + num_validation)
    x_val = x_train[val_mask]
    y_val = y_train[val_mask]

    train_mask = range(num_training)
    x_train = x_train[train_mask]
    y_train = y_train[train_mask]

    test_mask = range(num_test)
    x_test = x_test[test_mask]
    y_test = y_test[test_mask]

    # Normalize the data: subtract the mean pixel and divide by std
    mean_pixel = x_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = x_train.std(axis=(0, 1, 2), keepdims=True)

    x_train = (x_train - mean_pixel) / std_pixel
    x_val = (x_val - mean_pixel) / std_pixel
    x_test = (x_test - mean_pixel) / std_pixel

    return x_train, y_train, x_val, y_val, x_test, y_test


def run(**kwargs_dict):
    print(kwargs_dict)

    x_train, y_train, x_val, y_val, x_test, y_test = load_cifar10()

    print('train data shape: ', x_train.shape)
    print('train labels shape: ', y_train.shape, y_train.dtype)
    print('validation data shape: ', x_val.shape)
    print('validation labels shape: ', y_val.shape)
    print('test data shape: ', x_test.shape)
    print('test labels shape: ', y_test.shape)

    train_dataset = Dataset(x_train, y_train, batch_size=256, shuffle=True)
    val_dataset = Dataset(x_val, y_val, batch_size=256, shuffle=False)
    test_dataset = Dataset(x_test, y_test, batch_size=256)

    while True:
        # learning_rate = 0.0002
        learning_rate = 0.000630
        # learning_rate = 10 ** np.random.uniform(-3, -5)

        accuracy = train(
            '/device:GPU:0',
            train_dataset,
            val_dataset,
            custom_cnn,
            tf.train.AdamOptimizer(learning_rate),
            outer_loop_iterations=15000,
        )
        print(f"learning_rate: {learning_rate} \t| accuracy: {accuracy}")


###################################################################################################

from rcall import meta
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('backend')
    parser.add_argument('--cluster', type=str, default='ibis', help='name of cluster')

    args = parser.parse_args()
    kwargs_dict = vars(args)

    if args.backend == 'interactive':
        run(**kwargs_dict)
    else:
        meta.call(
            backend='kube',
            cluster=args.cluster,
            fn=run,
            kwargs=kwargs_dict,
            job_name='cs231n-cifar10',
            log_relpath=f'hp-{datetime.now().strftime("%Y%m%d%H%M")}',
            num_cpu='1',
            num_gpu='8',
            mpi_machines=1,
            mpi_proc_per_machine='num_gpu',
        )


if __name__ == '__main__':
    main()
