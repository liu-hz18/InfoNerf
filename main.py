from InfoNeRF import *
import os
import sys
import logging
import random
import argparse
import imageio
import numpy as np
from tqdm import tqdm

import jittor as jt
jt.flags.use_cuda = jt.has_cuda


def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_nerf(args):
    """
    Instantiate NeRF's MLP model.
    """
    # 位置信息的 embedder
    # multires = 10
    # embed_fn: output shape = [3*(1 + 10)] = [33] (input_ch)
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    # 方向信息的 embedder
    input_ch_views = 0
    # multires = 4
    # embed_fn: output shape = [2*(1 + 4)] = [10] (input_ch_views)
    embeddirs_fn, input_ch_views = get_embedder(
        args.multires_views, args.i_embed)

    skips = [4]
    if args.alpha_model_path is None:
        # 训练新模型
        # 第4层再次引入x（位置信息），最后输出(RGB\alpha) 4维
        model = NeRF(D=args.netdepth, W=args.netwidth,
                     input_ch=input_ch, skips=skips, input_ch_views=input_ch_views)
        grad_vars = list(model.parameters())
    else:
        # 加载checkpoint
        alpha_model = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                           input_ch=input_ch, skips=skips,
                           input_ch_views=input_ch_views)
        logger.info(f'Alpha model reloading from {args.alpha_model_path}')
        ckpt = jt.load(args.alpha_model_path)
        alpha_model.load_state_dict(ckpt['network_fine_state_dict'])
        # 使用checkpoint计算alpha, 使用新网络训练RGB
        if not args.no_coarse:
            model = NeRF_RGB(D=args.netdepth, W=args.netwidth,
                             input_ch=input_ch, skips=skips,
                             input_ch_views=input_ch_views, alpha_model=alpha_model)
            grad_vars = list(model.parameters())
        else:  # 直接使用保存的模型
            model = None
            grad_vars = []

    model_fine = None
    if args.N_importance > 0:
        if args.alpha_model_path is None:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                              input_ch=input_ch, skips=skips, input_ch_views=input_ch_views)
        else:
            model_fine = NeRF_RGB(D=args.netdepth_fine, W=args.netwidth_fine,
                                  input_ch=input_ch, skips=skips,
                                  input_ch_views=input_ch_views, alpha_model=alpha_model)
        grad_vars += list(model_fine.parameters())

    def network_query_fn(inputs, viewdirs, network_fn): return run_network(inputs, viewdirs, network_fn,
                                                                           embed_fn=embed_fn,
                                                                           embeddirs_fn=embeddirs_fn,
                                                                           netchunk=args.netchunk)

    # Create optimizer
    optimizer = jt.nn.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################
    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(
            os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    logger.info(f'Found ckpts {ckpts}')
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        if args.ckpt_render_iter is not None:
            ckpt_path = os.path.join(os.path.join(
                basedir, expname, f'{args.ckpt_render_iter:06d}.tar'))

        logger.info(f'Reloading from {ckpt_path}')
        ckpt = jt.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,  # 模型
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'entropy_ray_zvals': args.entropy,
        'extract_alpha': args.smoothing
    }

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train
    }
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    ##########################

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def train(args):
    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    render_first_time = True
    if args.render_pass:
        render_first_time = False

    ########################################
    #              Blender                 #
    ########################################
    if args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.testskip)
        # image: [N1, H, W, C=4], N1个图像，HxW大小，C: RGBalpha
        # render_poses: [N2, M1, M2], N2个图像，[M1, M2]是pose matrix, 包含rotation matrix 和 translation vector, 就是平移和旋转，是一般意义上的位姿矩阵T （camera-to-world affine）
        # hwf: 图像的高height, 宽度width, 和相机的焦距Focal
        logger.info(
            f'Loaded blender image={images.shape}, poses={poses.shape}, render_poses={render_poses.shape}, hwf={hwf}, dir={args.datadir}')
        i_train, i_val, i_test = i_split
        near = 2.
        far = 6.

        if args.fewshot > 0:
            if args.train_scene is None:
                np.random.seed(args.fewshot_seed)
                i_train = np.random.choice(
                    i_train, args.fewshot, replace=False)
            else:
                i_train = np.array(args.train_scene)
            logger.info(f'i_train {i_train}')

        if args.white_bkgd:
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
        else:
            images = images[..., :3]

    else:
        logger.info(f'Unknown dataset type {args.dataset_type}, exiting...')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    if args.render_test:
        render_poses = np.array(poses[i_test])
    elif args.render_train:
        render_poses = np.array(poses[i_train])

    # Short circuit if only rendering out from trained model
    if args.render_only:
        logger.info('RENDER ONLY')
        with jt.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            if args.render_test:
                if args.render_test_full:
                    testsavedir = os.path.join(
                        basedir, expname, 'full_renderonly_{}_{:06d}'.format('test', start))
                else:
                    testsavedir = os.path.join(
                        basedir, expname, 'renderonly_{}_{:06d}'.format('test', start))
            elif args.render_train:
                testsavedir = os.path.join(
                    basedir, expname, 'renderonly_{}_{:06d}'.format('train', start))
            else:
                testsavedir = os.path.join(
                    basedir, expname, 'renderonly_{}_{:06d}'.format('path', start))
            os.makedirs(testsavedir, exist_ok=True)
            logger.info(f'test poses shape {render_poses.shape}')

            rgbs, disps = render_path(render_poses, hwf, args.chunk, render_kwargs_test,
                                      gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            logger.info(f'Done rendering {testsavedir}')
            imageio.mimwrite(os.path.join(testsavedir, 'rgb.mp4'),
                             to8b(rgbs), fps=30, quality=8)
            disps[np.isnan(disps)] = 0
            logger.info(
                f'Depth stats {np.mean(disps)}, {np.max(disps)}, {np.percentile(disps, 95)}')
            imageio.mimwrite(os.path.join(testsavedir, 'disp.mp4'), to8b(
                disps / np.percentile(disps, 95)), fps=30, quality=8)
            return

    # Prepare raybatch tensor if batching random rays
    N_rgb = args.N_rand  # default: 1024

    if args.entropy:
        N_entropy = args.N_entropy  # default: 1024
        fun_entropy_loss = EntropyLoss(args)

    if args.smoothing:
        get_near_c2w = GetNearC2W(args)
        fun_KL_divergence_loss = SmoothingLoss(args)

    use_batching = not args.no_batching

    if use_batching:
        # For random ray batching
        logger.info('get rays')
        rays = np.stack([get_rays_np(H, W, focal, p)
                        for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        logger.info('done, concats')
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None]], 1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_all = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_all[i]
                            for i in i_train], 0)  # train images only
        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        logger.info('shuffle rays')
        np.random.shuffle(rays_rgb)

        rays_entropy = None
        if args.entropy:
            rays_entropy = np.stack(rays_all, 0)  # train images only
            # [(N-1)*H*W, ro+rd+rgb, 3]
            rays_entropy = np.reshape(rays_entropy, [-1, 3, 3])
            rays_entropy = rays_entropy.astype(np.float32)
            np.random.shuffle(rays_entropy)

    if use_batching:
        raysRGB_iter = iter(RayDataset(
            rays_rgb, batch_size=N_rgb, shuffle=True))
        raysEntropy_iter = iter(RayDataset(
            rays_entropy, batch_size=N_entropy, shuffle=True)) if rays_entropy is not None else None

    N_iters = args.N_iters + 1
    logger.info('Begin')
    logger.info(f'TRAIN views are {i_train}')  # [8 72 37 41]
    # [113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137]
    logger.info(f'TEST views are {i_test}')
    # [100 101 102 103 104 105 106 107 108 109 110 111 112]
    logger.info(f'VAL views are {i_val}')

    start = start + 1

    if args.eval_only:
        N_iters = start + 2

    logging_info = {
        "lr": optimizer.lr,
        "rgb_loss": np.nan,
        "psnr": np.nan,
        "entropy_ray_zvals": np.nan,
        "KL_loss": np.nan,
        "rgb0_loss": np.nan,
        "psnr0": np.nan,
    }
    pbar = tqdm(range(start, N_iters), desc="Training", postfix=logging_info)
    for i in pbar:
        # Sample random ray batch
        if use_batching:
            # Random over all images
            try:
                batch = next(raysRGB_iter)
            except StopIteration:
                raysRGB_iter = iter(RayDataset(
                    rays_rgb, batch_size=N_rgb, shuffle=True))
                batch = next(raysRGB_iter)
            batch = jt.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            if args.entropy and (args.N_entropy != 0):
                try:
                    batch_entropy = next(raysEntropy_iter)
                except StopIteration:
                    raysEntropy_iter = iter(RayDataset(
                        rays_entropy, batch_size=N_entropy, shuffle=True))
                    batch_entropy = next(raysEntropy_iter)
                batch_rays_entropy = jt.transpose(batch_entropy, 0, 1)[:2]

        else:
            # Random from one image
            img_i = np.random.choice(i_train) # return 1 sample
            target = images[img_i]
            rgb_pose = jt.float32(poses[img_i, :3, :4])

            if args.N_rand is not None:
                # translate camera frames's [o]rigin and [d]irection to world frame's
                # camera frames中, rays_d 有一维是-1.
                rays_o, rays_d = get_rays(H, W, focal, rgb_pose)  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    # precrop_frac(default=0.5): fraction of img taken for central crops
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = jt.stack(
                        jt.meshgrid(
                            jt.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                            jt.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1
                    )
                    if i == start:
                        logger.info(
                            f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = jt.stack(
                        jt.meshgrid(jt.linspace(0, H-1, H),
                                    jt.linspace(0, W-1, W)),
                        -1
                    )  # (H, W, 2)

                coords = jt.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(
                    coords.shape[0], size=[N_rgb], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = jt.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)
                target_s = jt.float32(target[select_coords[:, 0],
                                  select_coords[:, 1]])  # (N_rand, 3)
                if args.smoothing:
                    rgb_near_pose = get_near_c2w(rgb_pose, iter_=i)
                    near_rays_o, near_rays_d = get_rays(H, W, focal, jt.float32(rgb_near_pose))  # (H, W, 3), (H, W, 3)
                    near_rays_o = near_rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    near_rays_d = near_rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    near_batch_rays = jt.stack([near_rays_o, near_rays_d], 0) # (2, N_rand, 3)

            ########################################################
            #            Sampling for unseen rays                  #
            ########################################################
            # logger.info("Sampling for unseen rays")
            if args.entropy and (args.N_entropy != 0):
                img_i = np.random.choice(len(images))
                target = images[img_i]
                pose = jt.float32(poses[img_i, :3, :4])
                rays_o, rays_d = get_rays(H, W, focal, pose)  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = jt.stack(
                        jt.meshgrid(
                            jt.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                            jt.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1
                    )
                    if i == start:
                        logger.info(
                            f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                else:
                    if args.smooth_sampling_method == 'near_pixel':
                        padding = args.smooth_pixel_range
                        coords = jt.stack(
                                    jt.meshgrid(jt.linspace(padding, H-1+padding, H), jt.linspace(padding, W-1+padding, W)), 
                                    -1
                                 )  # (H, W, 2)
                    else:
                        coords = jt.stack(jt.meshgrid(jt.linspace(0, H-1, H), jt.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = jt.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(
                    coords.shape[0], size=[N_entropy], replace=False)  # (N_entropy,)
                select_coords = coords[select_inds].long()  # (N_entropy, 2)
                rays_o_ent = rays_o[select_coords[:, 0],
                                    select_coords[:, 1]]  # (N_entropy, 3)
                rays_d_ent = rays_d[select_coords[:, 0],
                                    select_coords[:, 1]]  # (N_entropy, 3)
                batch_rays_entropy = jt.stack(
                    [rays_o_ent, rays_d_ent], 0)  # (2, N_entropy, 3)

                ########################################################
                #   Ray sampling for information gain reduction loss   #
                ########################################################

                if args.smoothing:
                    if args.smooth_sampling_method == 'near_pixel':
                        near_select_coords = get_near_pixel(select_coords, args.smooth_pixel_range)
                        ent_near_rays_o = rays_o[near_select_coords[:, 0], near_select_coords[:, 1]]  # (N_rand, 3)
                        ent_near_rays_d = rays_d[near_select_coords[:, 0], near_select_coords[:, 1]]  # (N_rand, 3)
                        ent_near_batch_rays = jt.stack([ent_near_rays_o, ent_near_rays_d], 0) # (2, N_rand, 3)
                    elif args.smooth_sampling_method == 'near_pose':
                        ent_near_pose = get_near_c2w(pose, iter_=i)
                        ent_near_rays_o, ent_near_rays_d = get_rays(H, W, focal, jt.float32(ent_near_pose))  # (H, W, 3), (H, W, 3)
                        ent_near_rays_o = ent_near_rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                        ent_near_rays_d = ent_near_rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                        ent_near_batch_rays = jt.stack([ent_near_rays_o, ent_near_rays_d], 0) # (2, N_rand, 3)

        N_rgb = batch_rays.shape[1]

        if args.entropy and (args.N_entropy != 0):
            batch_rays = jt.concat([batch_rays, batch_rays_entropy], 1)

        if args.smoothing:
            if args.entropy and (args.N_entropy != 0):
                batch_rays = jt.concat([batch_rays, near_batch_rays, ent_near_batch_rays], 1)
            else: 
                batch_rays = jt.concat([batch_rays, near_batch_rays], 1)

        # Volumetric rendering. train process
        rgb, disp, acc, depth, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, retraw=True,
                                               **render_kwargs_train)

        if args.entropy:
            acc_raw = acc
            alpha_raw = extras['alpha']

        extras = {x: extras[x][:N_rgb] for x in extras}

        rgb = rgb[:N_rgb, :]
        disp = disp[:N_rgb]
        acc = acc[:N_rgb]

        optimizer.zero_grad()

        img_loss = img2mse(rgb, target_s)
        logging_info['rgb_loss'] = img_loss.item()
        entropy_ray_zvals_loss = 0
        smoothing_loss = 0

        ########################################################
        #            Ray Entropy Minimiation Loss              #
        ########################################################
        # logger.info("Ray Entropy Minimiation Loss")
        if args.entropy:
            # plog(p) for Ray Density
            entropy_ray_zvals_loss = fun_entropy_loss.ray_zvals(
                alpha_raw, acc_raw)
            logging_info['entropy_ray_zvals'] = entropy_ray_zvals_loss.item()

        if args.entropy_end_iter is not None:
            if i > args.entropy_end_iter:
                entropy_ray_zvals_loss = 0

        ########################################################
        #           Infomation Gain Reduction Loss             #
        ########################################################
        # logger.info("Infomation Gain Reduction Loss")
        smoothing_lambda = args.smoothing_lambda * args.smoothing_rate ** (int(i/args.smoothing_step_size))
        
        if args.smoothing:
            smoothing_loss = fun_KL_divergence_loss(alpha_raw)
            logging_info['KL_loss'] = smoothing_loss
            if args.smoothing_end_iter is not None:
                if i > args.smoothing_end_iter:
                    smoothing_loss = 0

        loss = img_loss + args.entropy_ray_zvals_lambda * entropy_ray_zvals_loss + smoothing_lambda * smoothing_loss
        with jt.no_grad():
            psnr = mse2psnr(img_loss)
        logging_info['psnr'] = psnr.item()

        if 'rgb0' in extras and not args.no_coarse:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            with jt.no_grad():
                psnr0 = mse2psnr(img_loss0)
            logging_info['rgb0_loss'] = img_loss0.item()
            logging_info['psnr0'] = psnr0.item()

        optimizer.step(loss)

        logging_info["lr"] = optimizer.lr

        pbar.set_postfix(logging_info)

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        # logger.info("update learning rate")
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            jt.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict() if render_kwargs_train['network_fn'] is not None else None,
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict() if render_kwargs_train['network_fine'] is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            logger.info(f'Saved checkpoints at {path}')

        if (args.i_video > 0 and i % args.i_video == 0 and i > 0):
            logger.info("generating videos...")
            # Turn on testing mode
            if render_first_time == False:
                render_first_time = True
                continue
            with jt.no_grad():
                rgbs, disps = render_path(
                    render_poses, hwf, args.chunk, render_kwargs_test)
            logger.info(f'Done, saving {rgbs.shape} {disps.shape}')
            moviebase = os.path.join(
                basedir, expname, '{}_spiral_{:06d}_'.format(expname.split('/')[-1], i))
            imageio.mimwrite(moviebase + 'rgb.mp4',
                             to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4',
                             to8b(disps / np.nanmax(disps)), fps=30, quality=8)

        if (i % args.i_testset == 0) and (i > 0) and (len(i_test) > 0):
            logger.info("testing...")
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            logger.info(f'test poses shape {poses[i_test].shape}')
            with jt.no_grad():
                rgbs, disps = render_path(poses[i_test], hwf, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
                logger.info(f'Saved test set')
                rgbs_b, images_b = jt.float32(rgbs), jt.float32(images[i_test])
                test_loss = img2mse(rgbs_b, images_b)
                test_psnr = mse2psnr(test_loss)
                test_redefine_psnr = img2psnr_redefine(rgbs_b, images_b)
            logger.info(
                f"[TEST] Iter: {i} Loss: {test_loss.item()}  PSNR: {test_psnr.item()} redefine_PSNR: {test_redefine_psnr.item()}")

        if i % args.i_print == 0:
            logger.info(
                f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        global_step += 1


def parse_args():
    parser = argparse.ArgumentParser(
        prog="InfoNeRF implemenation using Jittor")
    parser.add_argument("--expname", type=str,
                        default="lego", help='experiment name')
    parser.add_argument("--basedir", type=str, default="./log",
                        help="log and checkpoints directory")
    parser.add_argument("--datadir", type=str, default='./data/lego',
                        help='input data directory')
    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help="layers in network")  # 网络的深度（层数）
    parser.add_argument("--netwidth", type=int, default=256,
                        help="channels per layer")  # 网络的宽度，也就是每一层的神经元个数
    parser.add_argument("--netdepth_fine", type=int,
                        default=8, help="layers in fine network")
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=1024,
                        help='batch size (number of random rays per gradient step)')  # batch_size，光束的数量
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=500,
                        help='exponential learning rate decay (in 1000 steps)')  # 指数学习率衰减（1000 步）
    parser.add_argument("--chunk", type=int, default=1024*8,
                        help='number of rays processed in parallel, decrease if running out of memory')  # 并行处理的光线数量，如果内存不足则减少
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')  # 通过网络并行发送的点数，如果内存不足则减少
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')  # 一次只能从 1 张图像中获取随机光线
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')  # 不要从保存的 ckpt 重新加载权重
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')  # 为粗略网络重新加载特定权重 npy 文件
    #######################################################
    #         Ray Entropy Minimization Loss               #
    #######################################################
    parser.add_argument("--N_entropy", type=int, default=1024,
                        help='number of entropy ray')
    parser.add_argument("--entropy", action='store_true',
                        help='using entropy ray loss')
    parser.add_argument("--entropy_log_scaling", action='store_true',
                        help='using log scaling for entropy loss')
    parser.add_argument("--entropy_ignore_smoothing", action='store_true',
                        help='ignoring entropy for ray for smoothing')
    parser.add_argument("--entropy_end_iter", type=int, default=None,
                        help='end iteratio of entropy')
    parser.add_argument("--entropy_type", type=str, default='log2', choices=['log2', '1-p'],
                        help='choosing type of entropy')
    parser.add_argument("--entropy_acc_threshold", type=float, default=0.1,
                        help='threshold for acc masking')
    parser.add_argument("--computing_entropy_all", action='store_true',
                        help='computing entropy for both seen and unseen ')
    parser.add_argument("--entropy_ray_lambda", type=float, default=0.001,
                        help='entropy lambda for ray entropy loss')
    parser.add_argument("--entropy_ray_zvals_lambda", type=float, default=0.001,
                        help='entropy lambda for ray zvals entropy loss')
    
    #######################################################
    #         Infomation Gain Reduction Loss              #
    #######################################################
    parser.add_argument("--smoothing", action='store_true',
                        help='using information gain reduction loss')
    # choosing between rotating camera pose & near pixel
    parser.add_argument("--smooth_sampling_method", type=str, default='near_pose', 
        help='how to sample the near rays, near_pose: modifying camera pose, near_pixel: sample near pixel', 
                    choices=['near_pose', 'near_pixel'])
    # 1) sampling by rotating camera pose
    parser.add_argument("--near_c2w_type", type=str, default='rot_from_origin', 
                        help='random augmentation method')
    parser.add_argument("--near_c2w_rot", type=float, default=5, 
                        help='random augmentation rotate: degree')
    parser.add_argument("--near_c2w_trans", type=float, default=0.1, 
                        help='random augmentation translation')
    # 2) sampling with near pixel
    parser.add_argument("--smooth_pixel_range", type=int,
                        help='the maximum distance between the near ray & the original ray (pixel dimension)')
    # optimizing 
    parser.add_argument("--smoothing_lambda", type=float, default=0.00001, 
                        help='lambda for smoothing loss')
    parser.add_argument("--smoothing_activation", type=str, default='norm', choices=['norm', 'softmax'],
                        help='how to make alpha to the distribution')
    parser.add_argument("--smoothing_step_size", type=int, default=2500,
                        help='reducing smoothing every')
    parser.add_argument("--smoothing_rate", type=float, default=1,
                        help='reducing smoothing rate')
    parser.add_argument("--smoothing_end_iter", type=int, default=None,
                        help='when smoothing will be end')
    #######################################################
    #                      Others                         #
    #######################################################
    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')  # 每条射线的粗样本数
    parser.add_argument("--N_importance", type=int, default=128,
                        help='number of additional fine samples per ray')  # 每条射线的附加精细样本数
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')  # 设置为 0. 无抖动，1. 抖动
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')  # 为默认位置编码设置 0，为无设置 -1
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')  # 多分辨率。 位置编码的最大频率的 log2（3D 位置）
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')  # 位置编码的最大频率的 log2（2D 方向）
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')  # 噪音方差

    # rendering options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')  # 不要优化，重新加载权重和渲染 render_poses 路径
    parser.add_argument("--eval_only", action='store_true',
                        help='do not optimize, reload weights and evaluation and logging to wandb')  # 渲染测试集而不是 render_poses 路径
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_test_full", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--ckpt_render_iter", type=int, default=None,
                        help='checkpoint iteration')
    parser.add_argument("--render_test_ray", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_train", action='store_true',
                        help='render the train set instead of render_poses path')
    parser.add_argument("--render_mypath", action='store_true',
                        help='render the test path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')  # 下采样因子以加快渲染速度，设置为 4 或 8 用于快速预览
    parser.add_argument("--render_pass", action='store_true',
                        help='do not rendering when resume')
    # training options
    parser.add_argument("--precrop_iters", type=int, default=500,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float, default=.5,
                        help='fraction of img taken for central crops')
    # dataset options
    parser.add_argument("--dataset_type", type=str, default='blender',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')  # 将从测试/验证集中加载 1/N 图像，这对于像 deepvoxels 这样的大型数据集很有用
    parser.add_argument("--fewshot", type=int, default=4,
                        help='if 0 not using fewshot, else: using fewshot')
    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1000,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=10000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=10000,
                        help='frequency of render_poses video saving')
    # debug
    parser.add_argument("--debug", action='store_true')

    # new experiment by kangle
    parser.add_argument("--N_iters", type=int, default=50002,
                        help='number of iters')
    parser.add_argument("--alpha_model_path", type=str, default=None,
                        help='predefined alpha model')
    parser.add_argument("--no_coarse", action='store_true',
                        help="Remove coarse network.")
    parser.add_argument("--train_scene", nargs='+', type=int,
                        help='id of scenes used to train')
    parser.add_argument("--test_scene", nargs='+', type=int,
                        help='id of scenes used to test')

    return parser.parse_args()


def init_logger(args):
    logger = logging.getLogger("default")
    cmd_handler = logging.StreamHandler(sys.stdout)
    cmd_handler.setLevel(logging.DEBUG)
    cmd_handler.setFormatter(logging.Formatter(
        r"[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)s] %(message)s"))
    log_handler = logging.FileHandler(os.path.join(
        args.basedir, args.expname, "train.log"), mode="w+", encoding="utf-8")
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter(
        r"[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)s] %(message)s"))
    logger.addHandler(cmd_handler)
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)
    return logger


if __name__ == '__main__':
    args = parse_args()
    set_all_seed(0)
    os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
    logger = init_logger(args)
    train(args)
