import jittor as jt


######################################################
#           Ray Entropy Minimization Loss            #
######################################################
class EntropyLoss(object):

    def __init__(self, args):
        super(EntropyLoss, self).__init__()
        self.N_samples = args.N_rand
        self.type_ = args.entropy_type
        self.threshold = args.entropy_acc_threshold
        self.computing_entropy_all = args.computing_entropy_all
        self.smoothing = args.smoothing
        self.computing_ignore_smoothing = args.entropy_ignore_smoothing
        self.entropy_log_scaling = args.entropy_log_scaling
        self.N_entropy = args.N_entropy

        if self.N_entropy == 0:
            self.computing_entropy_all = True

    def ray_zvals(self, sigma, acc):
        if self.smoothing and self.computing_ignore_smoothing:
            N_smooth = sigma.size(0)//2
            acc = acc[:N_smooth]
            sigma = sigma[:N_smooth]
        # the entropy of a discrete ray density function
        if not self.computing_entropy_all:
            acc = acc[self.N_samples:]
            sigma = sigma[self.N_samples:]
        ray_prob = sigma / (jt.sum(sigma, -1).unsqueeze(-1)+1e-10)
        entropy_ray = self.entropy(ray_prob)
        entropy_ray_loss = jt.sum(entropy_ray, -1)

        # Disregarding non-hitting rays
        mask = (acc > self.threshold).detach()
        entropy_ray_loss *= mask
        if self.entropy_log_scaling:
            return jt.log(jt.mean(entropy_ray_loss) + 1e-10)
        return jt.mean(entropy_ray_loss)

    def entropy(self, prob):
        if self.type_ == 'log2':
            return -1*prob*jt.log2(prob+1e-10)
        elif self.type_ == '1-p':
            return prob*jt.log2(1-prob)


######################################################
#          Infomation Gain Reduction Loss            #
######################################################
class SmoothingLoss(object):
    def __init__(self, args):
        super(SmoothingLoss, self).__init__()
        self.smoothing_activation = args.smoothing_activation
        self.criterion = jt.nn.KLDivLoss(reduction='batchmean')
    
    def __call__(self, sigma):
        half_num = sigma.size(0)//2
        sigma_1= sigma[:half_num]
        sigma_2 = sigma[half_num:]

        if self.smoothing_activation == 'softmax':
            p = jt.nn.softmax(sigma_1, -1)
            q = jt.nn.softmax(sigma_2, -1)
        elif self.smoothing_activation == 'norm':
            p = sigma_1 / (jt.sum(sigma_1, -1,  keepdim=True) + 1e-10) + 1e-10
            q = sigma_2 / (jt.sum(sigma_2, -1, keepdim=True) + 1e-10) +1e-10
        loss = self.criterion(p.log(), q)
        return loss

