import os
import numpy as np
import jittor as jt
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio

LOG10 = jt.log(jt.float32([10.]))
# Miscs
def img2mse(x, y): return jt.mean((x - y) ** 2)
def mse2psnr(x): return -10. * jt.log(x) / LOG10
def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


def img2psnr_redefine(x, y):
    '''
    we redefine the PSNR function,
    [previous]
    average MSE -> PSNR(average MSE)

    [new]
    average PSNR(each image pair)
    '''
    image_num = x.size(0)
    mses = ((x-y)**2).reshape(image_num, -1).mean(-1)

    psnrs = [mse2psnr(mse) for mse in mses]
    psnr = jt.stack(psnrs).mean()
    return psnr


# Ray helpers
def get_rays(H, W, focal, c2w, padding=None):
    # jittor's meshgrid has indexing='ij'
    if padding is not None:
        i, j = jt.meshgrid(jt.linspace(-padding, W-1+padding, W+2*padding),
                           jt.linspace(-padding, H-1+padding, H+2*padding))
    else:
        i, j = jt.meshgrid(jt.linspace(0, W-1, W), jt.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    # 3rd dimension is -1
    dirs = jt.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -jt.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = jt.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(
        H, dtype=np.float32), indexing='xy')  # i: H x W, j: H x W
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -
                    np.ones_like(i)], -1)  # dirs: H x W x 3
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))  # H x W x 3
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
# 为了提高体素渲染效率
# 提出一个由粗到细的结构来训练体素网。首先采样一组位置信息，基于stratied sampling，然后训练一个“粗”网络。在此基础上，再训练一个"细"网络
def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans

    pdf = weights / jt.sum(weights, dim=1, keepdims=True)
    cdf = jt.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = jt.concat([jt.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = jt.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = jt.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = jt.searchsorted(cdf, u, right=True)
    below = jt.maximum(jt.zeros_like(inds-1), inds-1)
    above = jt.minimum((cdf.shape[-1]-1) * jt.ones_like(inds), inds)
    inds_g = jt.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = jt.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = jt.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = jt.where(denom < 1e-5, jt.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def visualize_sigma(sigma, z_vals, filename):
    plt.plot(z_vals, sigma)
    plt.xlabel('z_vals')
    plt.ylabel('sigma')
    plt.savefig(filename)


def batchify(fn, chunk):
    if chunk is None:
        return fn

    def ret(inputs):
        return jt.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    inputs_flat = jt.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = jt.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = jt.concat([embedded, embedded_dirs], -1)
    # 把 embedded 分batch输入到网络中, 再把结果拼接起来
    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = jt.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


# 根据 render_pose 进行渲染，得到 N 个视角的图像，然后再合成 mp4
def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []
    accs = []

    for i, c2w in enumerate(tqdm(render_poses)):
        c2w = jt.float32(c2w)
        rgb, disp, acc, depth, extras = render(
            H, W, focal, chunk=chunk, c2w=c2w[:3, :4], retraw=True, **render_kwargs)
        rgbs.append(rgb.numpy())
        disps.append(disp.numpy())
        accs.append(acc.numpy())

        if savedir is not None:
            rgb8 = to8b(rgb.numpy())
            rgb8[np.isnan(rgb8)] = 0
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            depth = depth.numpy()
            depth = (np.clip(depth / 5, 0, 1) * 255.).astype(np.uint8)
            imageio.imwrite(os.path.join(
                savedir, '{:03d}_depth.png'.format(i)), depth)

        del rgb
        del disp
        del acc
        del extras
        del depth
        jt.sync_all()
        jt.gc()

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    return rgbs, disps


# 通过 体渲染公式 得到 指定视角下的渲染结果
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False,
                out_alpha=False, out_sigma=False, out_dist=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map. 差异图
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    # calculate \sum_{i=1}^{N} T_i (1 - exp(-sigma_i * delta_i)) c_i
    sigma_fn = jt.nn.relu

    ################################
    # 计算 `alpha` = 1 - exp(-sigma_i * delta_i)
    def raw2alpha(raw, dists, act_fn=sigma_fn): return 1. - \
        jt.exp(-act_fn(raw)*dists)

    # 相邻两个样本的距离 delta_i = t_{i+1} - t_i
    dists = z_vals[..., 1:] - z_vals[..., :-1]  # [N_rays, N_samples-1]
    dists = jt.concat([dists, jt.float32([1e10]).expand(
        dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    # [N_rays, N_samples]
    dists = dists * jt.norm(rays_d[..., None, :], dim=-1)

    noise = 0.
    if raw_noise_std > 0.:
        noise = jt.randn(raw[..., 3].shape) * raw_noise_std
    # Ray Density `alpha` = 1 - exp(-sigma_i * delta_i)
    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    ################################

    # 计算: T_i = exp(-\sum_{j=1}^{i-1}sigma_j * delta_j) = \cumprod_{j=1}^{i-1} exp(-sigma_j * delta_j) = \cumprod_{j=1}^{i-1} (1.0 - alpha_j)
    # equals to: weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * jt.cumprod(
        jt.concat([jt.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1),
        dim=-1
    )[:, :-1]  # [N_rays, N_samples]
    # 计算 c_i
    rgb = jt.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    # 得到最终的积分，即 渲染结果的颜色
    rgb_map = jt.sum(weights[..., None] * rgb, dim=-2)  # [N_rays, 3]
    # 到物体的距离
    depth_map = jt.sum(weights * z_vals, dim=-1)  # [N_rays]
    # Sum of weights along each ray. ??? TODO: why not `jt.sum(alpha, -1)`
    acc_map = jt.sum(weights, -1)  # [num_rays]
    # Disparity map. Inverse of depth map.
    disp_map = 1./jt.maximum(jt.float32(1e-10) *
                             jt.ones_like(depth_map), depth_map / acc_map)  # [N_rays]

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[..., None])

    others = {}
    if out_alpha or out_sigma or out_dist:
        if out_alpha:
            others['alpha'] = alpha
        if out_sigma:
            others['sigma'] = sigma_fn(raw[..., 3] + noise)
        if out_dist:
            others['dists'] = dists
        return rgb_map, disp_map, acc_map, weights, depth_map, others
    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    retraw=False,
    lindisp=False,
    perturb=0.,
    N_importance=0,
    network_fine=None,
    white_bkgd=False,
    raw_noise_std=0.,
    entropy_ray_zvals=None,
    extract_xyz=None,
    extract_alpha=None,
    extract_sigma=None,
):
    """
    Volumetric rendering.
    Args:
        ray_batch: array of shape [batch_size, ...]. All information necessary for sampling along a ray, including: ray origin, ray direction, min dist, max dist, and unit-magnitude viewing direction.
        network_fn: function. Model for predicting RGB and density at each point in space. 用于预测每个点的 RGB 和密度的模型
        network_query_fn : function used for passing queries to network_fn.
        N_samples: int. Number of different times to sample along each ray. 每条射线上的采样次数
        retraw: bool. If True, include model’s raw, unprocessed predictions.
        lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
        perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified random points in time. 1 则每条射线都分层采样
        N_importance: int. Number of additional times to sample along each ray. 每条射线上的额外采样数 These samples are only passed to network_fine.
        network_fine: “fine” network with same spec as network_fn.
        white_bkgd: bool. If True, assume a white background.
        raw_noise_std: ...
        verbose: bool. If True, print more debugging info.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
        disp_map: [num_rays]. Disparity map. 1 / depth.
        acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
        raw: [num_rays, num_samples, 4]. Raw predictions from model.
        rgb0: See rgb_map. Output for coarse model.
        disp0: See disp_map. Output for coarse model.
        acc0: See acc_map. Output for coarse model.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 9 else None
    bounds = jt.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # 在每个光束上, 取 N_sample 个点
    t_vals = jt.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    # [N_rays, N_samples], 这一个batch的所有光束的采样点
    z_vals = z_vals.expand([N_rays, N_samples])
    # 上面的N_sample个点是等距的，这里在每两个点之间的区间中随机选一个点，这样就有了新的均匀抽样的N_sample个点，引入了随机性
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = jt.concat([mids, z_vals[..., -1:]], -1)
        lower = jt.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = jt.rand(z_vals.shape)

        z_vals = lower + (upper - lower) * t_rand
    # 得到空间中的N_sample个点的具体位置
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3]

    # forwarding: raw shape = [N_rays, N_samples, 4] (RGBalpha)
    if network_fn is not None:
        # 我们不需要调用 model.train(), 因为模型中不含随机层比如dropout等
        raw = network_query_fn(pts, viewdirs, network_fn)
    else:
        if network_fine.alpha_model is not None:
            raw = network_query_fn(pts, viewdirs, network_fine.alpha_model)
        else:
            raw = network_query_fn(pts, viewdirs, network_fine)
    # 通过 体渲染公式 得到 指定视角下的渲染结果
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    # Hierarchical sampling
    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        # 重新采样光束上的点
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = z_samples.detach()

        z_vals = jt.float32(
            np.sort(jt.concat([z_vals, z_samples], -1).numpy(), axis=-1))
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        if entropy_ray_zvals or extract_sigma or extract_alpha:
            rgb_map, disp_map, acc_map, weights, depth_map, others = raw2outputs(
                raw, z_vals, rays_d, raw_noise_std, white_bkgd, out_sigma=True, out_alpha=True, out_dist=True)
        else:
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
                raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map,
           'acc_map': acc_map, 'depth_map': depth_map}

    if entropy_ray_zvals or extract_sigma or extract_alpha:
        ret['sigma'] = others['sigma']
        ret['alpha'] = others['alpha']
        ret['z_vals'] = z_vals
        ret['dists'] = others['dists']

    if extract_xyz:
        ret['xyz'] = jt.sum(weights.unsqueeze(-1)*pts, -2)

    if retraw:
        ret['raw'] = raw

    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0

    return ret


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: jt.concat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, chunk=1024*32, rays=None, near=0., far=1.,
           use_viewdirs=False, depths=None, c2w=None,
           **kwargs):
    """Render rays"""
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        viewdirs = viewdirs / jt.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = jt.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = jt.reshape(rays_o, [-1, 3]).float()
    rays_d = jt.reshape(rays_d, [-1, 3]).float()

    near, far = near * \
        jt.ones_like(rays_d[..., :1]), far * jt.ones_like(rays_d[..., :1])
    rays = jt.concat([rays_o, rays_d, near, far], -1)  # B x 8
    if depths is not None:
        rays = jt.concat([rays, depths.reshape(-1, 1)], -1)
    if use_viewdirs:
        rays = jt.concat([rays, viewdirs], -1)
    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = jt.reshape(all_ret[k], k_sh)

    # 返回: 光束对应的rgb, 视差图, 不透明度
    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]
