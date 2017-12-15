import pickle
from model import *
from utils import *
from config import config, log_config
from scipy.io import loadmat, savemat


def main_train():
    mask_perc = tl.global_flag['maskperc']
    mask_name = tl.global_flag['mask']
    model_name = tl.global_flag['model']

    # =================================== BASIC CONFIGS =================================== #

    print('[*] run basic configs ... ')

    log_dir = "log_{}_{}_{}".format(model_name, mask_name, mask_perc)
    tl.files.exists_or_mkdir(log_dir)
    log_all, log_eval, log_50, log_all_filename, log_eval_filename, log_50_filename = logging_setup(log_dir)

    checkpoint_dir = "checkpoint_{}_{}_{}".format(model_name, mask_name, mask_perc)
    tl.files.exists_or_mkdir(checkpoint_dir)

    save_dir = "samples_{}_{}_{}".format(model_name, mask_name, mask_perc)
    tl.files.exists_or_mkdir(save_dir)

    # configs
    batch_size = config.TRAIN.batch_size
    early_stopping_num = config.TRAIN.early_stopping_num
    g_alpha = config.TRAIN.g_alpha
    g_beta = config.TRAIN.g_beta
    g_gamma = config.TRAIN.g_gamma
    g_adv = config.TRAIN.g_adv
    lr = config.TRAIN.lr
    lr_decay = config.TRAIN.lr_decay
    decay_every = config.TRAIN.decay_every
    beta1 = config.TRAIN.beta1
    n_epoch = config.TRAIN.n_epoch
    sample_size = config.TRAIN.sample_size

    log_config(log_all_filename, config)
    log_config(log_eval_filename, config)
    log_config(log_50_filename, config)

    # ==================================== PREPARE DATA ==================================== #

    print('[*] load data ... ')
    training_data_path = config.TRAIN.training_data_path
    val_data_path = config.TRAIN.val_data_path
    testing_data_path = config.TRAIN.testing_data_path

    with open(training_data_path, 'rb') as f:
        X_train = pickle.load(f)

    with open(val_data_path, 'rb') as f:
        X_val = pickle.load(f)

    with open(testing_data_path, 'rb') as f:
        X_test = pickle.load(f)

    print('X_train shape/min/max: ', X_train.shape, X_train.min(), X_train.max())
    print('X_val shape/min/max: ', X_val.shape, X_val.min(), X_val.max())
    print('X_test shape/min/max: ', X_test.shape, X_test.min(), X_test.max())

    print('[*] loading mask ... ')
    if mask_name == "gaussian2d":
        mask = \
            loadmat(
                os.path.join(config.TRAIN.mask_Gaussian2D_path, "GaussianDistribution2DMask_{}.mat".format(mask_perc)))[
                'maskRS2']
    elif mask_name == "gaussian1d":
        mask = \
            loadmat(
                os.path.join(config.TRAIN.mask_Gaussian1D_path, "GaussianDistribution1DMask_{}.mat".format(mask_perc)))[
                'maskRS1']
    elif mask_name == "poisson2d":
        mask = \
            loadmat(
                os.path.join(config.TRAIN.mask_Gaussian1D_path, "PoissonDistributionMask_{}.mat".format(mask_perc)))[
                'population_matrix']
    else:
        raise ValueError("no such mask exists: {}".format(mask_name))

    # ==================================== DEFINE MODEL ==================================== #

    print('[*] define model ... ')

    nw, nh, nz = X_train.shape[1:]

    # define placeholders
    t_image_good = tf.placeholder('float32', [batch_size, nw, nh, nz], name='good_image')
    t_image_good_samples = tf.placeholder('float32', [sample_size, nw, nh, nz], name='good_image_samples')
    t_image_bad = tf.placeholder('float32', [batch_size, nw, nh, nz], name='bad_image')
    t_image_bad_samples = tf.placeholder('float32', [sample_size, nw, nh, nz], name='bad_image_samples')
    t_gen = tf.placeholder('float32', [batch_size, nw, nh, nz], name='generated_image_for_test')
    t_gen_sample = tf.placeholder('float32', [sample_size, nw, nh, nz], name='generated_sample_image_for_test')
    t_image_good_244 = tf.placeholder('float32', [batch_size, 244, 244, 3], name='vgg_good_image')

    # define generator network
    if tl.global_flag['model'] == 'unet':
        net = u_net_bn(t_image_bad, is_train=True, reuse=False, is_refine=False)
        net_test = u_net_bn(t_image_bad, is_train=False, reuse=True, is_refine=False)
        net_test_sample = u_net_bn(t_image_bad_samples, is_train=False, reuse=True, is_refine=False)

    elif tl.global_flag['model'] == 'unet_refine':
        net = u_net_bn(t_image_bad, is_train=True, reuse=False, is_refine=True)
        net_test = u_net_bn(t_image_bad, is_train=False, reuse=True, is_refine=True)
        net_test_sample = u_net_bn(t_image_bad_samples, is_train=False, reuse=True, is_refine=True)
    else:
        raise Exception("unknown model")

    # define discriminator network
    net_d, logits_fake = discriminator(net.outputs, is_train=True, reuse=False)
    _, logits_real = discriminator(t_image_good, is_train=True, reuse=True)

    # define VGG network
    net_vgg_conv4_good, _ = vgg16_cnn_emb(t_image_good_244, reuse=False)
    net_vgg_conv4_gen, _ = vgg16_cnn_emb(tf.tile(tf.image.resize_images(net.outputs, [244, 244]), [1, 1, 1, 3]), reuse=True)

    # ==================================== DEFINE LOSS ==================================== #

    print('[*] define loss functions ... ')

    # discriminator loss
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    d_loss = d_loss1 + d_loss2

    # generator loss (adversarial)
    g_loss = tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')

    # generator loss (perceptual)
    g_perceptual = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(
        net_vgg_conv4_good.outputs,
        net_vgg_conv4_gen.outputs),
        axis=[1, 2, 3]))

    # generator loss (pixel-wise)
    g_nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(net.outputs, t_image_good), axis=[1, 2, 3]))
    g_nmse_b = tf.sqrt(tf.reduce_sum(tf.square(t_image_good), axis=[1, 2, 3]))
    g_nmse = tf.reduce_mean(g_nmse_a / g_nmse_b)

    # generator loss (frequency)
    fft_good_abs = tf.map_fn(fft_abs_for_map_fn, t_image_good)
    fft_gen_abs = tf.map_fn(fft_abs_for_map_fn, net.outputs)
    g_fft = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(fft_good_abs, fft_gen_abs), axis=[1, 2]))

    # generator loss (total)
    g_loss = g_adv * g_loss + g_alpha * g_nmse + g_gamma * g_perceptual + g_beta * g_fft

    # nmse metric for testing purpose
    nmse_a_0_1 = tf.sqrt(tf.reduce_sum(tf.squared_difference(t_gen, t_image_good), axis=[1, 2, 3]))
    nmse_b_0_1 = tf.sqrt(tf.reduce_sum(tf.square(t_image_good), axis=[1, 2, 3]))
    nmse_0_1 = nmse_a_0_1 / nmse_b_0_1

    nmse_a_0_1_sample = tf.sqrt(tf.reduce_sum(tf.squared_difference(t_gen_sample, t_image_good_samples), axis=[1, 2, 3]))
    nmse_b_0_1_sample = tf.sqrt(tf.reduce_sum(tf.square(t_image_good_samples), axis=[1, 2, 3]))
    nmse_0_1_sample = nmse_a_0_1_sample / nmse_b_0_1_sample

    # ==================================== DEFINE TRAIN OPTS ==================================== #

    print('[*] define training options ... ')

    g_vars = tl.layers.get_variables_with_name('u_net', True, True)
    d_vars = tl.layers.get_variables_with_name('discriminator', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)

    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    # ==================================== TRAINING ==================================== #

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tl.layers.initialize_global_variables(sess)

    # load generator and discriminator weights (for continuous training purpose)
    tl.files.load_and_assign_npz(sess=sess,
                                 name=os.path.join(checkpoint_dir, tl.global_flag['model']) + '.npz',
                                 network=net)
    tl.files.load_and_assign_npz(sess=sess,
                                 name=os.path.join(checkpoint_dir, tl.global_flag['model']) + '_d.npz',
                                 network=net_d)

    # load vgg weights
    net_vgg_conv4_path = config.TRAIN.VGG16_path
    npz = np.load(net_vgg_conv4_path)
    assign_op = []
    for idx, val in enumerate(sorted(npz.items())[0:20]):
        print("  Loading pretrained VGG16, CNN part %s" % str(val[1].shape))
        assign_op.append(net_vgg_conv4_good.all_params[idx].assign(val[1]))
    sess.run(assign_op)
    net_vgg_conv4_good.print_params(False)

    n_training_examples = len(X_train)
    n_step_epoch = round(n_training_examples / batch_size)

    # sample testing images
    idex = tl.utils.get_random_int(min=0, max=len(X_test) - 1, number=sample_size, seed=config.TRAIN.seed)
    X_samples_good = X_test[idex]
    X_samples_bad = threading_data(X_samples_good, fn=to_bad_img, mask=mask)

    x_good_sample_rescaled = (X_samples_good + 1) / 2
    x_bad_sample_rescaled = (X_samples_bad + 1) / 2

    tl.visualize.save_images(X_samples_good,
                             [5, 10],
                             os.path.join(save_dir, "sample_image_good.png"))

    tl.visualize.save_images(X_samples_bad,
                             [5, 10],
                             os.path.join(save_dir, "sample_image_bad.png"))

    tl.visualize.save_images(np.abs(X_samples_good - X_samples_bad),
                             [5, 10],
                             os.path.join(save_dir, "sample_image_diff_abs.png"))

    tl.visualize.save_images(np.sqrt(np.abs(X_samples_good - X_samples_bad) / 2 + config.TRAIN.epsilon),
                             [5, 10],
                             os.path.join(save_dir, "sample_image_diff_sqrt_abs.png"))

    tl.visualize.save_images(np.clip(10 * np.abs(X_samples_good - X_samples_bad) / 2, 0, 1),
                             [5, 10],
                             os.path.join(save_dir, "sample_image_diff_sqrt_abs_10_clip.png"))

    tl.visualize.save_images(threading_data(X_samples_good, fn=distort_img),
                             [5, 10],
                             os.path.join(save_dir, "sample_image_aug.png"))
    scipy.misc.imsave(os.path.join(save_dir, "mask.png"), mask * 255)

    print('[*] start training ... ')

    best_nmse = np.inf
    best_epoch = 1
    esn = early_stopping_num
    for epoch in range(0, n_epoch):

        # learning rate decay
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr * new_lr_decay))
            log = " ** new learning rate: %f" % (lr * new_lr_decay)
            print(log)
            log_all.debug(log)
        elif epoch == 0:
            log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            print(log)
            log_all.debug(log)

        for step in range(n_step_epoch):
            step_time = time.time()
            idex = tl.utils.get_random_int(min=0, max=n_training_examples - 1, number=batch_size)
            X_good = X_train[idex]
            X_good_aug = threading_data(X_good, fn=distort_img)
            X_good_244 = threading_data(X_good_aug, fn=vgg_prepro)
            X_bad = threading_data(X_good_aug, fn=to_bad_img, mask=mask)

            errD, _ = sess.run([d_loss, d_optim], {t_image_good: X_good_aug, t_image_bad: X_bad})
            errG, errG_perceptual, errG_nmse, errG_fft, _ = sess.run([g_loss, g_perceptual, g_nmse, g_fft, g_optim],
                                                                     {t_image_good_244: X_good_244,
                                                                      t_image_good: X_good_aug,
                                                                      t_image_bad: X_bad})

            log = "Epoch[{:3}/{:3}] step={:3} d_loss={:5} g_loss={:5} g_perceptual_loss={:5} g_mse={:5} g_freq={:5} took {:3}s".format(
                epoch + 1,
                n_epoch,
                step,
                round(float(errD), 3),
                round(float(errG), 3),
                round(float(errG_perceptual), 3),
                round(float(errG_nmse), 3),
                round(float(errG_fft), 3),
                round(time.time() - step_time, 2))

            print(log)
            log_all.debug(log)

        # evaluation for training data
        total_nmse_training = 0
        total_ssim_training = 0
        total_psnr_training = 0
        num_training_temp = 0
        for batch in tl.iterate.minibatches(inputs=X_train, targets=X_train, batch_size=batch_size, shuffle=False):
            x_good, _ = batch
            # x_bad = threading_data(x_good, fn=to_bad_img, mask=mask)
            x_bad = threading_data(
                x_good,
                fn=to_bad_img,
                mask=mask)

            x_gen = sess.run(net_test.outputs, {t_image_bad: x_bad})

            x_good_0_1 = (x_good + 1) / 2
            x_gen_0_1 = (x_gen + 1) / 2

            nmse_res = sess.run(nmse_0_1, {t_gen: x_gen_0_1, t_image_good: x_good_0_1})
            ssim_res = threading_data([_ for _ in zip(x_good_0_1, x_gen_0_1)], fn=ssim)
            psnr_res = threading_data([_ for _ in zip(x_good_0_1, x_gen_0_1)], fn=psnr)
            total_nmse_training += np.sum(nmse_res)
            total_ssim_training += np.sum(ssim_res)
            total_psnr_training += np.sum(psnr_res)
            num_training_temp += batch_size

        total_nmse_training /= num_training_temp
        total_ssim_training /= num_training_temp
        total_psnr_training /= num_training_temp

        log = "Epoch: {}\nNMSE training: {:8}, SSIM training: {:8}, PSNR training: {:8}".format(
            epoch + 1,
            total_nmse_training,
            total_ssim_training,
            total_psnr_training)
        print(log)
        log_all.debug(log)
        log_eval.info(log)

        # evaluation for validation data
        total_nmse_val = 0
        total_ssim_val = 0
        total_psnr_val = 0
        num_val_temp = 0
        for batch in tl.iterate.minibatches(inputs=X_val, targets=X_val, batch_size=batch_size, shuffle=False):
            x_good, _ = batch
            # x_bad = threading_data(x_good, fn=to_bad_img, mask=mask)
            x_bad = threading_data(
                x_good,
                fn=to_bad_img,
                mask=mask)

            x_gen = sess.run(net_test.outputs, {t_image_bad: x_bad})

            x_good_0_1 = (x_good + 1) / 2
            x_gen_0_1 = (x_gen + 1) / 2

            nmse_res = sess.run(nmse_0_1, {t_gen: x_gen_0_1, t_image_good: x_good_0_1})
            ssim_res = threading_data([_ for _ in zip(x_good_0_1, x_gen_0_1)], fn=ssim)
            psnr_res = threading_data([_ for _ in zip(x_good_0_1, x_gen_0_1)], fn=psnr)
            total_nmse_val += np.sum(nmse_res)
            total_ssim_val += np.sum(ssim_res)
            total_psnr_val += np.sum(psnr_res)
            num_val_temp += batch_size

        total_nmse_val /= num_val_temp
        total_ssim_val /= num_val_temp
        total_psnr_val /= num_val_temp

        log = "Epoch: {}\nNMSE val: {:8}, SSIM val: {:8}, PSNR val: {:8}".format(
            epoch + 1,
            total_nmse_val,
            total_ssim_val,
            total_psnr_val)
        print(log)
        log_all.debug(log)
        log_eval.info(log)

        img = sess.run(net_test_sample.outputs, {t_image_bad_samples: X_samples_bad})
        tl.visualize.save_images(img,
                                 [5, 10],
                                 os.path.join(save_dir, "image_{}.png".format(epoch)))

        if total_nmse_val < best_nmse:
            esn = early_stopping_num  # reset early stopping num
            best_nmse = total_nmse_val
            best_epoch = epoch + 1

            # save current best model
            tl.files.save_npz(net.all_params,
                              name=os.path.join(checkpoint_dir, tl.global_flag['model']) + '.npz',
                              sess=sess)

            tl.files.save_npz(net_d.all_params,
                              name=os.path.join(checkpoint_dir, tl.global_flag['model']) + '_d.npz',
                              sess=sess)
            print("[*] Save checkpoints SUCCESS!")
        else:
            esn -= 1

        log = "Best NMSE result: {} at {} epoch".format(best_nmse, best_epoch)
        log_eval.info(log)
        log_all.debug(log)
        print(log)

        # early stopping triggered
        if esn == 0:
            log_eval.info(log)

            tl.files.load_and_assign_npz(sess=sess,
                                         name=os.path.join(checkpoint_dir, tl.global_flag['model']) + '.npz',
                                         network=net)
            # evluation for test data
            x_gen = sess.run(net_test_sample.outputs, {t_image_bad_samples: X_samples_bad})
            x_gen_0_1 = (x_gen + 1) / 2
            savemat(save_dir + '/test_random_50_generated.mat', {'x_gen_0_1': x_gen_0_1})

            nmse_res = sess.run(nmse_0_1_sample, {t_gen_sample: x_gen_0_1, t_image_good_samples: x_good_sample_rescaled})
            ssim_res = threading_data([_ for _ in zip(x_good_sample_rescaled, x_gen_0_1)], fn=ssim)
            psnr_res = threading_data([_ for _ in zip(x_good_sample_rescaled, x_gen_0_1)], fn=psnr)

            log = "NMSE testing: {}\nSSIM testing: {}\nPSNR testing: {}\n\n".format(
                nmse_res,
                ssim_res,
                psnr_res)

            log_50.debug(log)

            log = "NMSE testing average: {}\nSSIM testing average: {}\nPSNR testing average: {}\n\n".format(
                np.mean(nmse_res),
                np.mean(ssim_res),
                np.mean(psnr_res))

            log_50.debug(log)

            log = "NMSE testing std: {}\nSSIM testing std: {}\nPSNR testing std: {}\n\n".format(np.std(nmse_res),
                                                                                                np.std(ssim_res),
                                                                                                np.std(psnr_res))

            log_50.debug(log)

            # evaluation for zero-filled (ZF) data
            nmse_res_zf = sess.run(nmse_0_1_sample,
                                   {t_gen_sample: x_bad_sample_rescaled, t_image_good_samples: x_good_sample_rescaled})
            ssim_res_zf = threading_data([_ for _ in zip(x_good_sample_rescaled, x_bad_sample_rescaled)], fn=ssim)
            psnr_res_zf = threading_data([_ for _ in zip(x_good_sample_rescaled, x_bad_sample_rescaled)], fn=psnr)

            log = "NMSE ZF testing: {}\nSSIM ZF testing: {}\nPSNR ZF testing: {}\n\n".format(
                nmse_res_zf,
                ssim_res_zf,
                psnr_res_zf)

            log_50.debug(log)

            log = "NMSE ZF average testing: {}\nSSIM ZF average testing: {}\nPSNR ZF average testing: {}\n\n".format(
                np.mean(nmse_res_zf),
                np.mean(ssim_res_zf),
                np.mean(psnr_res_zf))

            log_50.debug(log)

            log = "NMSE ZF std testing: {}\nSSIM ZF std testing: {}\nPSNR ZF std testing: {}\n\n".format(
                np.std(nmse_res_zf),
                np.std(ssim_res_zf),
                np.std(psnr_res_zf))

            log_50.debug(log)

            # sample testing images
            tl.visualize.save_images(x_gen,
                                     [5, 10],
                                     os.path.join(save_dir, "final_generated_image.png"))

            tl.visualize.save_images(np.clip(10 * np.abs(X_samples_good - x_gen) / 2, 0, 1),
                                     [5, 10],
                                     os.path.join(save_dir, "final_generated_image_diff_abs_10_clip.png"))

            tl.visualize.save_images(np.clip(10 * np.abs(X_samples_good - X_samples_bad) / 2, 0, 1),
                                     [5, 10],
                                     os.path.join(save_dir, "final_bad_image_diff_abs_10_clip.png"))

            print("[*] Job finished!")
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='unet', help='unet, unet_refine')
    parser.add_argument('--mask', type=str, default='gaussian2d', help='gaussian1d, gaussian2d, poisson2d')
    parser.add_argument('--maskperc', type=int, default='30', help='10,20,30,40,50')

    args = parser.parse_args()

    tl.global_flag['model'] = args.model
    tl.global_flag['mask'] = args.mask
    tl.global_flag['maskperc'] = args.maskperc

    main_train()
