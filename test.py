import pickle
from model import *
from utils import *
from config import config, log_config
from scipy.io import loadmat, savemat


def main_test():
    mask_perc = tl.global_flag['maskperc']
    mask_name = tl.global_flag['mask']
    model_name = tl.global_flag['model']

    # =================================== BASIC CONFIGS =================================== #

    print('[*] run basic configs ... ')

    log_dir = "log_inference_{}_{}_{}".format(model_name, mask_name, mask_perc)
    tl.files.exists_or_mkdir(log_dir)
    _, _, log_inference, _, _, log_inference_filename = logging_setup(log_dir)

    checkpoint_dir = "checkpoint_inference_{}_{}_{}".format(model_name, mask_name, mask_perc)
    tl.files.exists_or_mkdir(checkpoint_dir)

    save_dir = "samples_inference_{}_{}_{}".format(model_name, mask_name, mask_perc)
    tl.files.exists_or_mkdir(save_dir)

    # configs
    sample_size = config.TRAIN.sample_size

    # ==================================== PREPARE DATA ==================================== #

    print('[*] load data ... ')
    testing_data_path = config.TRAIN.testing_data_path

    with open(testing_data_path, 'rb') as f:
        X_test = pickle.load(f)

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

    nw, nh, nz = X_test.shape[1:]

    # define placeholders
    t_image_good = tf.placeholder('float32', [sample_size, nw, nh, nz], name='good_image')     
    t_image_bad = tf.placeholder('float32', [sample_size, nw, nh, nz], name='bad_image')
    t_gen = tf.placeholder('float32', [sample_size, nw, nh, nz], name='generated_image')

    # define generator network
    if tl.global_flag['model'] == 'unet':
        net_test = u_net_bn(t_image_bad, is_train=False, reuse=False, is_refine=False)
    elif tl.global_flag['model'] == 'unet_refine':
        net_test = u_net_bn(t_image_bad, is_train=False, reuse=False, is_refine=True)
    else:
        raise Exception("unknown model")

    # nmse metric for testing purpose
    nmse_a_0_1 = tf.sqrt(tf.reduce_sum(tf.squared_difference(t_gen, t_image_good), axis=[1, 2, 3]))
    nmse_b_0_1 = tf.sqrt(tf.reduce_sum(tf.square(t_image_good), axis=[1, 2, 3]))
    nmse_0_1 = nmse_a_0_1 / nmse_b_0_1

    # ==================================== INFERENCE ==================================== #

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tl.files.load_and_assign_npz(sess=sess,
                                 name=os.path.join(checkpoint_dir, tl.global_flag['model']) + '.npz',
                                 network=net_test)

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

    print('[*] start testing ... ')

    x_gen = sess.run(net_test.outputs, {t_image_bad: X_samples_bad})
    x_gen_0_1 = (x_gen + 1) / 2

    # evaluation for generated data

    nmse_res = sess.run(nmse_0_1, {t_gen: x_gen_0_1, t_image_good: x_good_sample_rescaled})
    ssim_res = threading_data([_ for _ in zip(x_good_sample_rescaled, x_gen_0_1)], fn=ssim)
    psnr_res = threading_data([_ for _ in zip(x_good_sample_rescaled, x_gen_0_1)], fn=psnr)

    log = "NMSE testing: {}\nSSIM testing: {}\nPSNR testing: {}\n\n".format(
        nmse_res,
        ssim_res,
        psnr_res)

    log_inference.debug(log)

    log = "NMSE testing average: {}\nSSIM testing average: {}\nPSNR testing average: {}\n\n".format(
        np.mean(nmse_res),
        np.mean(ssim_res),
        np.mean(psnr_res))

    log_inference.debug(log)

    log = "NMSE testing std: {}\nSSIM testing std: {}\nPSNR testing std: {}\n\n".format(np.std(nmse_res),
                                                                                        np.std(ssim_res),
                                                                                        np.std(psnr_res))

    log_inference.debug(log)

    # evaluation for zero-filled (ZF) data
    nmse_res_zf = sess.run(nmse_0_1,
                           {t_gen: x_bad_sample_rescaled, t_image_good: x_good_sample_rescaled})
    ssim_res_zf = threading_data([_ for _ in zip(x_good_sample_rescaled, x_bad_sample_rescaled)], fn=ssim)
    psnr_res_zf = threading_data([_ for _ in zip(x_good_sample_rescaled, x_bad_sample_rescaled)], fn=psnr)

    log = "NMSE ZF testing: {}\nSSIM ZF testing: {}\nPSNR ZF testing: {}\n\n".format(
        nmse_res_zf,
        ssim_res_zf,
        psnr_res_zf)

    log_inference.debug(log)

    log = "NMSE ZF average testing: {}\nSSIM ZF average testing: {}\nPSNR ZF average testing: {}\n\n".format(
        np.mean(nmse_res_zf),
        np.mean(ssim_res_zf),
        np.mean(psnr_res_zf))

    log_inference.debug(log)

    log = "NMSE ZF std testing: {}\nSSIM ZF std testing: {}\nPSNR ZF std testing: {}\n\n".format(
        np.std(nmse_res_zf),
        np.std(ssim_res_zf),
        np.std(psnr_res_zf))

    log_inference.debug(log)

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

    main_test()
