from tensorlayer.layers import *
from utils import *


def discriminator(input_images, is_train=True, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(input_images,
                            name='input')

        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                        padding='SAME', W_init=w_init, name='h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim * 2, (4, 4), (2, 2), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h1/batchnorm')

        net_h2 = Conv2d(net_h1, df_dim * 4, (4, 4), (2, 2), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h2/batchnorm')

        net_h3 = Conv2d(net_h2, df_dim * 8, (4, 4), (2, 2), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h3/batchnorm')

        net_h4 = Conv2d(net_h3, df_dim * 16, (4, 4), (2, 2), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h4/conv2d')
        net_h4 = BatchNormLayer(net_h4, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h4/batchnorm')

        net_h5 = Conv2d(net_h4, df_dim * 32, (4, 4), (2, 2), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h5/conv2d')
        net_h5 = BatchNormLayer(net_h5, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h5/batchnorm')

        net_h6 = Conv2d(net_h5, df_dim * 16, (1, 1), (1, 1), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h6/conv2d')
        net_h6 = BatchNormLayer(net_h6, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h6/batchnorm')

        net_h7 = Conv2d(net_h6, df_dim * 8, (1, 1), (1, 1), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h7/conv2d')
        net_h7 = BatchNormLayer(net_h7, is_train=is_train, gamma_init=gamma_init, name='h7/batchnorm')

        net = Conv2d(net_h7, df_dim * 2, (1, 1), (1, 1), act=None,
                     padding='SAME', W_init=w_init, b_init=b_init, name='h7_res/conv2d')
        net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
                             is_train=is_train, gamma_init=gamma_init, name='h7_res/batchnorm')
        net = Conv2d(net, df_dim * 2, (3, 3), (1, 1), act=None,
                     padding='SAME', W_init=w_init, b_init=b_init, name='h7_res/conv2d2')
        net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
                             is_train=is_train, gamma_init=gamma_init, name='h7_res/batchnorm2')
        net = Conv2d(net, df_dim * 8, (3, 3), (1, 1), act=None,
                     padding='SAME', W_init=w_init, b_init=b_init, name='h7_res/conv2d3')
        net = BatchNormLayer(net, is_train=is_train, gamma_init=gamma_init, name='h7_res/batchnorm3')

        net_h8 = ElementwiseLayer(layer=[net_h7, net], combine_fn=tf.add, name='h8/add')
        net_h8.outputs = tl.act.lrelu(net_h8.outputs, 0.2)

        net_ho = FlattenLayer(net_h8, name='output/flatten')
        net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity, W_init=w_init, name='output/dense')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)

    return net_ho, logits


def u_net_bn(x, is_train=False, reuse=False, is_refine=False):

    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name='input')

        conv1 = Conv2d(inputs, 64, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv1')
        conv2 = Conv2d(conv1, 128, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv2')
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bn2')

        conv3 = Conv2d(conv2, 256, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv3')
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bn3')

        conv4 = Conv2d(conv3, 512, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv4')
        conv4 = BatchNormLayer(conv4, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bn4')

        conv5 = Conv2d(conv4, 512, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv5')
        conv5 = BatchNormLayer(conv5, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bn5')

        conv6 = Conv2d(conv5, 512, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv6')
        conv6 = BatchNormLayer(conv6, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bn6')

        conv7 = Conv2d(conv6, 512, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv7')
        conv7 = BatchNormLayer(conv7, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bn7')

        conv8 = Conv2d(conv7, 512, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                       padding='SAME', W_init=w_init, b_init=b_init, name='conv8')

        up7 = DeConv2d(conv8, 512, (4, 4), out_size=(2, 2), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv7')
        up7 = BatchNormLayer(up7, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn7')

        up6 = ConcatLayer([up7, conv7], concat_dim=3, name='concat6')
        up6 = DeConv2d(up6, 1024, (4, 4), out_size=(4, 4), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv6')
        up6 = BatchNormLayer(up6, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn6')

        up5 = ConcatLayer([up6, conv6], concat_dim=3, name='concat5')
        up5 = DeConv2d(up5, 1024, (4, 4), out_size=(8, 8), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv5')
        up5 = BatchNormLayer(up5, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn5')

        up4 = ConcatLayer([up5, conv5], concat_dim=3, name='concat4')
        up4 = DeConv2d(up4, 1024, (4, 4), out_size=(16, 16), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv4')
        up4 = BatchNormLayer(up4, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn4')

        up3 = ConcatLayer([up4, conv4], concat_dim=3, name='concat3')
        up3 = DeConv2d(up3, 256, (4, 4), out_size=(32, 32), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv3')
        up3 = BatchNormLayer(up3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn3')

        up2 = ConcatLayer([up3, conv3], concat_dim=3, name='concat2')
        up2 = DeConv2d(up2, 128, (4, 4), out_size=(64, 64), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv2')
        up2 = BatchNormLayer(up2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn2')

        up1 = ConcatLayer([up2, conv2], concat_dim=3, name='concat1')
        up1 = DeConv2d(up1, 64, (4, 4), out_size=(128, 128), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv1')
        up1 = BatchNormLayer(up1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn1')

        up0 = ConcatLayer([up1, conv1], concat_dim=3, name='concat0')
        up0 = DeConv2d(up0, 64, (4, 4), out_size=(256, 256), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv0')
        up0 = BatchNormLayer(up0, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn0')

        if is_refine:
            out = Conv2d(up0, 1, (1, 1), act=tf.nn.tanh, name='out')
            out = ElementwiseLayer([out, inputs], tf.add, 'add_for_refine')
            out.outputs = tl.act.ramp(out.outputs, v_min=-1, v_max=1)
        else:
            out = Conv2d(up0, 1, (1, 1), act=tf.nn.tanh, name='out')

    return out


def vgg16_cnn_emb(t_image, reuse=False):
    with tf.variable_scope("vgg16_cnn", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        t_image = (t_image + 1) * 127.5  # convert input of [-1, 1] to [0, 255]

        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net_in = InputLayer(t_image - mean, name='vgg_input_im')

        # conv1
        network = tl.layers.Conv2dLayer(net_in,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 3, 64],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv1_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 64, 64],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv1_2')
        network = tl.layers.PoolLayer(network,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='vgg_pool1')

        # conv2
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 64, 128],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv2_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 128, 128],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv2_2')
        network = tl.layers.PoolLayer(network,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='vgg_pool2')

        # conv3
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 128, 256],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv3_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 256, 256],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv3_2')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 256, 256],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv3_3')
        network = tl.layers.PoolLayer(network,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='vgg_pool3')
        # conv4
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 256, 512],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv4_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv4_2')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv4_3')

        network = tl.layers.PoolLayer(network,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='vgg_pool4')
        conv4 = network

        # conv5
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv5_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv5_2')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv5_3')
        network = tl.layers.PoolLayer(network,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='vgg_pool5')

        network = FlattenLayer(network, name='vgg_flatten')

        return conv4, network


if __name__ == "__main__":
    pass
