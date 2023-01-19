import jittor as jt
from jittor import nn

# Positional encoding (section 5.1)
# 应用在位置(x,y,z)和方向(u, v)信息的每一个分量上
# 对坐标的变换取L=10, 对视角的变换取L=4.


class Embedder(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        freq_bands = 2.**jt.linspace(0., max_freq, steps=N_freqs)
        # gamma(p) = (sin(2^0 p), cos(2^0 p), sin(2^1 p), cos(2^1 p), ..., sin(2^{N_freqs-1} p), cos(2^{N_freqs-1} p))
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return jt.concat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'periodic_fns': [jt.sin, jt.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


# Model， 输出RGBalpha
class NeRF(nn.Module):

    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, skips=[4]):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W)
                                        if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)]
        )

        # Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + W, W//2)])

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)

    def execute(self, x):
        input_pts, input_views = jt.split(
            x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = nn.relu(h)
            if i in self.skips:
                h = jt.concat([input_pts, h], -1)
        # get alpha
        alpha = self.alpha_linear(h)
        # get rgb
        feature = self.feature_linear(h)
        h = jt.concat([feature, input_views], -1)
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = nn.relu(h)
        rgb = self.rgb_linear(h)
        # concat
        outputs = jt.concat([rgb, alpha], -1)
        # output shape = [N_rays, N_samples_from_each_ray, 4] (RGBalpha)
        return outputs


# 使用checkpoint得到alpha, 使用正在训练的模型得到RGB
class NeRF_RGB(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, skips=[4], alpha_model=None):
        super(NeRF_RGB, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W)
                                        if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)]
        )

        # Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + W, W//2)])

        self.feature_linear = nn.Linear(W, W)
        self.rgb_linear = nn.Linear(W//2, 3)

        self.alpha_model = alpha_model

    def execute(self, x):
        input_pts, input_views = jt.split(
            x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = nn.relu(h)
            if i in self.skips:
                h = jt.concat([input_pts, h], -1)
        # get alpha from checkpoints
        with jt.no_grad():
            # only extract "alpha" channel (a.k.a. sigma)
            alpha = self.alpha_model(x)[..., 3][..., None]
        # get rgb from trained model
        feature = self.feature_linear(h)
        h = jt.concat([feature, input_views], -1)
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = nn.relu(h)
        rgb = self.rgb_linear(h)
        # concat
        outputs = jt.concat([rgb, alpha], -1)

        return outputs
