import torch
import torch.nn.functional as F

# DCGAN loss


def loss_dcgan_dis(dis_fake, dis_real):
    L1 = torch.mean(F.softplus(-dis_real))
    L2 = torch.mean(F.softplus(dis_fake))
    return L1, L2


def loss_dcgan_gen(dis_fake):
    loss = torch.mean(F.softplus(-dis_fake))
    return loss

def loss_dcgan_dis_new(dis_fake, dis_real, dis_real_fake) :
    L1 = torch.mean(F.softplus(-dis_real))
    L2 = torch.mean(F.softplus(dis_fake))
    L_real_fake = torch.mean(F.softplus(dis_real_fake))
    return L1, L2, L_real_fake
# Hinge Loss
# def loss_hinge_dis(dis_fake, dis_real, ratio):
def loss_hinge_dis(dis_fake, dis_real):

    """
    fixed to take in density ratio estimates
    """
    # properly match up dimensions, and only reweight real examples
    # weighted = F.relu(1. - dis_real) * ratio.unsqueeze(1)
    weighted = F.relu(1. - dis_real) 
    loss_real = torch.mean(weighted)
    # loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
    # loss = torch.mean(F.relu(1. - dis_real))
    # loss += torch.mean(F.relu(1. + dis_fake))
    # return loss

def loss_hinge_analysis(dis_real):

    """
    fixed to take in density ratio estimates
    """
    # properly match up dimensions, and only reweight real examples
    # weighted = F.relu(1. - dis_real) * ratio.unsqueeze(1)
    weighted = F.relu(1. - dis_real)
    loss_real = weighted
    return loss_real

def loss_hinge_dis_new(dis_fake, dis_real, dis_real_fake):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    loss_real_fake = torch.mean(F.relu(1. + dis_real_fake))
    return loss_real, loss_fake, loss_real_fake

def loss_hinge_dis_new_fake(dis_fake, dis_real, dis_real_fake, dis_fake_fake):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    loss_real_fake = torch.mean(F.relu(1. + dis_real_fake))
    loss_fake_fake = torch.mean(F.relu(1. + dis_fake_fake))
    return loss_real, loss_fake, loss_real_fake, loss_fake_fake

def loss_hinge_gen(dis_fake):
    # with torch.no_grad():
    #   dis_fake_norm = torch.exp(dis_fake).mean()
    #   dis_fake_ratio = torch.exp(dis_fake) / dis_fake_norm
    # dis_fake = dis_fake * dis_fake_ratio
    loss = -torch.mean(dis_fake)
    return loss


def loss_kl_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    with torch.no_grad():
        dis_fake_m = dis_fake - dis_fake.mean()
        dis_fake_m = torch.clamp(dis_fake_m, min=-10.0, max=10.0)
        dis_fake_norm = torch.exp(dis_fake_m).mean() + 1e-8
        dis_fake_ratio = (torch.exp(dis_fake_m) + 1e-8) / dis_fake_norm
    dis_fake = dis_fake * dis_fake_ratio
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake

def loss_kl_dis_new(dis_fake, dis_real, dis_real_fake):
    loss_real = torch.mean(F.relu(1. - dis_real))
    with torch.no_grad():
        dis_fake_m = dis_fake - dis_fake.mean()
        dis_fake_m = torch.clamp(dis_fake_m, min=-10.0, max=10.0)
        dis_fake_norm = torch.exp(dis_fake_m).mean() + 1e-8
        dis_fake_ratio = (torch.exp(dis_fake_m) + 1e-8) / dis_fake_norm
    dis_fake = dis_fake * dis_fake_ratio
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    
    with torch.no_grad():
        dis_fake_m = dis_real_fake - dis_real_fake.mean()
        dis_fake_m = torch.clamp(dis_fake_m, min=-10.0, max=10.0)
        dis_fake_norm = torch.exp(dis_fake_m).mean() + 1e-8
        dis_fake_ratio = (torch.exp(dis_fake_m) + 1e-8) / dis_fake_norm
    dis_real_fake = dis_real_fake * dis_fake_ratio
    loss_real_fake = torch.mean(F.relu(1. + dis_real_fake))

    return loss_real, loss_fake, loss_real_fake


def loss_kl_gen(dis_fake):
    with torch.no_grad():
        dis_fake_m = dis_fake - dis_fake.mean()
        dis_fake_m = torch.clamp(dis_fake_m, min=-10.0, max=10.0)
        dis_fake_norm = torch.exp(dis_fake_m).mean() + 1e-8
        dis_fake_ratio = (torch.exp(dis_fake_m) + 1e-8) / dis_fake_norm
    dis_fake = dis_fake * dis_fake_ratio
    loss = -torch.mean(dis_fake)
    return loss


def loss_kl_grad_dis(dis_fake, dis_real):
    dis_fake_m = dis_fake - dis_fake.mean()
    dis_fake_m = torch.clamp(dis_fake_m, min=-10.0, max=10.0)
    dis_fake_norm = torch.exp(dis_fake_m).mean() + 1e-8
    dis_fake_ratio = (torch.exp(dis_fake_m) + 1e-8) / dis_fake_norm
    dis_fake = dis_fake * dis_fake_ratio

    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake


def loss_kl_grad_gen(dis_fake):
    dis_fake_m = dis_fake - dis_fake.mean()
    dis_fake_m = torch.clamp(dis_fake_m, min=-10.0, max=10.0)
    dis_fake_norm = torch.exp(dis_fake_m).mean() + 1e-8
    dis_fake_ratio = (torch.exp(dis_fake_m) + 1e-8) / dis_fake_norm
    dis_fake = dis_fake * dis_fake_ratio

    loss = -torch.mean(dis_fake)
    return loss

# maybe modify for this

# restricted kl-fgan


def loss_f_kl_dis(dis_fake, dis_real):
    import ipdb
    ipdb.set_trace()
    loss_real = torch.mean(F.relu(1.0 - dis_real))
    loss_fake = torch.mean(torch.exp(dis_fake - 1.0))
    return loss_real, loss_fake


def loss_f_kl_gen(dis_fake):
    import ipdb
    ipdb.set_trace()
    loss = -torch.mean(torch.exp(dis_fake - 1.0))
    return loss


def loss_dv_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1.0 - dis_real))
    loss_fake = -torch.logsumexp(dis_fake) / dis_fake.size(0)
    return loss_real, loss_fake


def loss_dv_gen(dis_fake):
    loss = torch.logsumexp(dis_fake) / dis_fake.size(0)
    return loss


# chi^2

def loss_chi_dis(dis_fake, dis_real):
    dis_fake = torch.clamp(dis_fake, -1.0, 1.0)
    dis_real = torch.clamp(dis_real, -1.0, 1.0)
    loss_real = torch.mean(- dis_real)
    dis_fake_mean = torch.mean(dis_fake)
    loss_fake = torch.mean(dis_fake * (dis_fake - dis_fake_mean + 2)) / 2.0
    return loss_real, loss_fake


def loss_chi_gen(dis_fake):
    dis_fake = torch.clamp(dis_fake, -1.0, 1.0)
    dis_fake_mean = torch.mean(dis_fake)
    loss_fake = -torch.mean(dis_fake * (dis_fake - dis_fake_mean + 2)) / 2.0
    return loss_fake


def loss_dv_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    dis_fake_norm = torch.exp(dis_fake).mean() + 1e-8
    dis_fake_ratio = (torch.exp(dis_fake) + 1e-8) / dis_fake_norm
    dis_fake = dis_fake * dis_fake_ratio
    loss_fake = torch.mean(F.relu(1. + dis_fake)) + \
        torch.mean(dis_fake_ratio * torch.log(dis_fake_ratio))
    return loss_real, loss_fake


def loss_dv_gen(dis_fake):
    dis_fake_norm = torch.exp(dis_fake).mean() + 1e-8
    dis_fake_ratio = (torch.exp(dis_fake) + 1e-8) / dis_fake_norm
    dis_fake = dis_fake * dis_fake_ratio
    loss = -torch.mean(dis_fake) - \
        torch.mean(dis_fake_ratio * torch.log(dis_fake_ratio))
    return loss


# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis
