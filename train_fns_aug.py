''' train_fns.py
Functions for the main loop of training different conditional image models
'''
import torch
import torch.nn as nn
import torchvision
import os

import utils
import losses
import numpy as np
import sys
from PIL import Image
import torchvision.transforms as transforms

# Dummy training function for debugging
def dummy_training_function():
    def train(x, y):
        return {}
    return train


def select_loss(config):
    if config['loss_type'] == 'hinge':
        return losses.loss_hinge_dis_new, losses.loss_hinge_gen
    elif config['loss_type'] == 'dcgan':
        return losses.loss_dcgan_dis_new, losses.loss_dcgan_gen
    elif config['loss_type'] == 'kl':
        return losses.loss_kl_dis_new, losses.loss_kl_gen
    elif config['loss_type'] == 'kl_gen':
        return losses.loss_hinge_dis, losses.loss_kl_gen
    elif config['loss_type'] == 'kl_dis':
        return losses.loss_kl_dis, losses.loss_hinge_gen
    elif config['loss_type'] == 'kl_grad':
        return losses.loss_kl_grad_dis, losses.loss_kl_grad_gen
    elif config['loss_type'] == 'f_kl':
        return losses.loss_f_kl_dis, losses.loss_f_kl_gen
    elif config['loss_type'] == 'chi2':
        return losses.loss_chi_dis, losses.loss_chi_gen
    elif config['loss_type'] == 'dv':
        return losses.loss_dv_dis, losses.loss_dv_gen
    else:
        raise ValueError('loss not defined')

def shifted(img) :
    min_shift = 10
    max_shift = 25
    shift = min_shift + (max_shift - min_shift) * torch.rand(1)
    k = shift.long().item()
    new_data = img.clone()
    new_data[:,:,:k,:k] = img[:,:,-k:,-k:]
    new_data[:,:,:k,k:] = img[:,:,-k:,:-k]
    new_data[:,:,k:,:k] = img[:,:,:-k,-k:]
    return new_data

def jigsaw_k(data, k = 4) :
    actual_h = data.size()[2]
    actual_w = data.size()[3]
    h = torch.split(data, int(actual_h/k), dim = 2)
    splits = []
    for i in range(k) :
        splits += torch.split(h[i], int(actual_w/k), dim = 3)
    fake_samples = torch.stack(splits, -1)
    for idx in range(fake_samples.size()[0]) :
        fake_samples[idx] = fake_samples[idx,:,:,:,torch.randperm(k*k)]
    fake_samples = torch.split(fake_samples, 1, dim=4)
    merged = []
    for i in range(k) :
        merged += [torch.cat(fake_samples[i*k:(i+1)*k], 2)]
    fake_samples = torch.squeeze(torch.cat(merged, 3), -1)
    return fake_samples

def rand_jigsaw(data) :
    K = [2,4]
    idx = (len(K)) * torch.rand(1)
    idx = idx.long().item()
    k = K[idx]
    return jigsaw_k(data, k)

def curriculum_jigsaw(data, epoch, config) :
    if epoch <= config['curriculum_epochs'] :
        k = 4
    else :
        k = 2 
    return jigsaw_k(data, k)

def permutation(data) :
    indices = torch.randperm(data.size(0))
    return data[indices]

def stitch_other_data(data, k = 2, other_data = None) :
    if other_data is None :
        data_perm = permutation(data)
    else :
        if torch.randint(0, 2, (1,))[0].item() == 0 :
            data_perm = other_data
        else :
            data_perm = permutation(data)
    actual_h = data.size()[2]
    actual_w = data.size()[3]
    if torch.randint(0, 2, (1,))[0].item() == 0 :
        dim0, dim1 = 2,3
    else :
        dim0, dim1 = 3,2

    h = torch.split(data, int(actual_h/k), dim = dim0)
    h_1 = torch.split(data_perm, int(actual_h/k), dim = dim0)
    splits = []
    for i in range(k) :
        if i < int(k/2) :
            splits += torch.split(h[i], int(actual_w/k), dim = dim1)
        else :
            splits += torch.split(h_1[i], int(actual_w/k), dim = dim1)
    merged = []
    for i in range(k) :
        merged += [torch.cat(splits[i*k:(i+1)*k], dim1)]
    fake_samples = torch.cat(merged, dim0)

    return fake_samples

def stitch(data, k = 2) :
    indices = torch.randperm(data.size(0))
    data_perm = data[indices]
    actual_h = data.size()[2]
    actual_w = data.size()[3]
    if torch.randint(0, 2, (1,))[0].item() == 0 :
        dim0, dim1 = 2,3
    else :
        dim0, dim1 = 3,2

    h = torch.split(data, int(actual_h/k), dim = dim0)
    h_1 = torch.split(data_perm, int(actual_h/k), dim = dim0)
    splits = []
    for i in range(k) :
        if i < int(k/2) :
            splits += torch.split(h[i], int(actual_w/k), dim = dim1)
        else :
            splits += torch.split(h_1[i], int(actual_w/k), dim = dim1)
    merged = []
    for i in range(k) :
        merged += [torch.cat(splits[i*k:(i+1)*k], dim1)]
    fake_samples = torch.cat(merged, dim0)

    return fake_samples
        
def mixup(data, alpha = 1.0) :
    lamb = np.random.beta(alpha, alpha)
    indices = torch.randperm(data.size(0))
    data_perm = data[indices]
    return data*lamb + (1-lamb)*data_perm
    
def noise(data) :
    return (torch.rand_like(data) - 0.5)/0.5


def rand_blur(data) :
    min_k, max_k = 10, 20
    data = data.clone()
    h, w = data.size(2), data.size(3)
    b_size = data.size(0)
    for i in range(b_size) :
        k = (min_k + (max_k - min_k) * torch.rand(1)).long().item()
        h_pos = ((h - k) * torch.rand(1)).long().item()
        w_pos = ((w - k) * torch.rand(1)).long().item()
        patch = data[i,:,h_pos:h_pos+k,w_pos:w_pos+k]
        patch_mean = torch.mean(patch, axis = (1,2), keepdim = True)
        data[i,:,h_pos:h_pos+k,w_pos:w_pos+k] = torch.ones_like(patch) * patch_mean

    return data
 

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cut_mix(data, beta = 1.0) :
    data = data.clone()
    lam = np.random.beta(beta, beta)
    #rand_index = torch.randperm(data.size()[0]).cuda()
    #data_perm = data[rand_index]
    data_perm = permutation(data)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data_perm[:, :, bbx1:bbx2, bby1:bby2] 
    return data

def rotate(data, angle = 60) :
    batch_size = data.size(0)
    new_data = []
    for i in range(batch_size) :
        pil_img = transforms.ToPILImage()(data[i].cpu())
        img_rotated = transforms.functional.rotate(pil_img, angle)
        new_data.append(transforms.ToTensor()(img_rotated))
    new_data = torch.stack(new_data, 0)
    return new_data.cuda()

def GAN_training_function(G, D, GD, z_, y_, ema, state_dict, config):
    discriminator_loss, generator_loss = select_loss(config)
            
        
    def train(x, y):
        G.optim.zero_grad()
        D.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, config['batch_size'])
        y = torch.split(y, config['batch_size'])
        counter = 0

        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, True)
            utils.toggle_grad(G, False)

        for step_index in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step
            D.optim.zero_grad()
            for accumulation_index in range(config['num_D_accumulations']):
                z_.sample_()
                if not config['conditional']:
                    # only feed in 0's for y if "unconditional"
                    y_.zero_()
                    y_counter = torch.zeros_like(y[counter]).to(y_.device).long()
                else:
                    y_.sample_()
                    y_counter = y[counter]
                
                real_samples = x[counter]
                real_fake_samples = jigsaw_k(real_samples, k = 2)  
                # real_fake_samples = stitch(real_samples, k = 2)
                # real_fake_samples = mixup(real_samples, alpha = 25.0)
                #real_fake_samples = rand_blur(real_samples)
                #real_fake_samples = cut_mix(real_samples)

                D_fake, D_real, D_real_fake = GD(z_[:config['batch_size']], y_[:config['batch_size']], real_samples, real_fake_samples,
                                    y_counter, train_G=False, split_D=config['split_D'])
                # y_.sample_()
                # D_fake, D_real = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                #                     x[counter], y[counter], train_G=False,
                #                     split_D=config['split_D'])
                # Compute components of D's loss, average them, and divide by
                # the number of gradient accumulations
                # if D_fake.max().item() - D_fake.min().item() > 30:
                # import ipdb
                # ipdb.set_trace()

                D_loss_real, D_loss_fake,D_loss_real_fake = discriminator_loss(
                    D_fake, D_real, D_real_fake)
                if config['curriculum_baseline'] and state_dict['epoch'] <= config['curriculum_epochs'] :
                    D_loss = (D_loss_real + D_loss_fake)/float(config['num_D_accumulations'])
                else : 
                    D_loss = (D_loss_real + config['alpha']*D_loss_fake + config['beta']*D_loss_real_fake) / \
                        float(config['num_D_accumulations'])

                D_loss.backward()
                counter += 1

            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D')
                utils.ortho(D, config['D_ortho'])

            D.optim.step()

        # Optionally toggle "requires_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(G, True)

        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()

        # If accumulating gradients, loop multiple times
        for accumulation_index in range(config['num_G_accumulations']):
            z_.sample_()
            y_.sample_()
            if not config['conditional']:
                y_.zero_()
            D_fake = GD(z_, y_, train_G=True, split_D=config['split_D'])
            G_loss = generator_loss(
                D_fake) / float(config['num_G_accumulations'])
            G_loss.backward()

        # Optionally apply modified ortho reg in G
        if config['G_ortho'] > 0.0:
            # Debug print to indicate we're using ortho reg in G
            print('using modified ortho reg in G')
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(G, config['G_ortho'],
                        blacklist=[param for param in G.shared.parameters()])
        G.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])

        out = {'G_loss': float(G_loss.item()),
               'D_loss_real': float(D_loss_real.item()),
               'D_loss_fake': float(D_loss_fake.item())}
        # Return G's loss and the components of D's loss.
        return out
    return train


''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''

def GAN_log_function(G, D, GD, z_, y_, ema, state_dict, config):
    import numpy as np
    import math
    discriminator_loss = losses.loss_hinge_analysis

    def train():

        corrupt_dict = {'elastic':'elastic_transform','jpeg':'jpeg_compression','speckle':'speckle_noise','gaussian':'gaussian_noise',
                        'blur':'gaussian_blur',
                        'zoom':'zoom_blur','brightness':'brightness', 'contrast':'contrast','defocus':'defocus_blur',
                        'fog':'fog','frost':'frost','glass':'glass_blur','impulse':'impulse_noise','motion':'motion_blur',
                        'pixelate':'pixelate','saturate':'saturate','shot':'shot_noise','snow':'snow','spatter':'spatter',
                        'train':'train_samples', 'test':'test_samples'
                        }
        base_dir = "../mintnet/CIFAR-10-C/"
        for mode in corrupt_dict :

            # test_batch = np.transpose(np.load("../mintnet/CIFAR-10-C/elastic_transform.npy"), (0, 3, 1, 2))[-10000:]
            test_batch = np.transpose(np.load(base_dir + corrupt_dict[mode]+".npy"), (0, 3, 1, 2))
            batch_size = 50
            test_iters = int(math.ceil(len(test_batch)/batch_size))
            logp = []
            y_counter = torch.zeros(batch_size).to("cuda").long()
            losses = []
            for idx in range(test_iters) :
                data_batch = test_batch[idx*batch_size:(idx+1)*batch_size]
                test_data = torch.from_numpy(data_batch).float()
                test_data = test_data.to("cuda")

                data = test_data/255.0

                D_fake, D_real = GD(z_[:config['batch_size']], y_[:config['batch_size']], data,
                                    y_counter, train_G=False, split_D=config['split_D'])

                #D_loss_real = discriminator_loss(
                #        D_real)

                #D_loss_real_np = np.squeeze(D_loss_real.cpu().data.numpy(), -1) 
                D_real_np = np.squeeze(D_real.cpu().data.numpy(), -1)

                logp.extend(D_real_np)
                #losses.extend(D_loss_real_np)
                print("%d iters reached"%idx)


            logp = np.array(logp)
            #losses = np.array(losses)
            # print(np.min(cleaned_data), np.max(cleaned_data), cleaned_data.shape)
            # np.save("corruption/elastic_pgd.npy", cleaned_data)
            print("logp calculation done for the corruption ", mode, " and the shape of data is ", logp.shape, " min is ", np.min(logp),
                            " max is ", np.max(logp))
            np.save("corruption_new/"+mode+"_logp.npy", logp)
            #np.save("corruption/temp.npy", losses)
            #break


    return train

def save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                    state_dict, config, experiment_name):
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name, None, G_ema if config['ema'] else None)
    # Save an additional copy to mitigate accidental corruption if process
    # is killed during a save (it's happened to me before -.-)
    if config['num_save_copies'] > 0:
        utils.save_weights(G, D, state_dict, config['weights_root'],
                           experiment_name,
                           'copy%d' % state_dict['save_num'],
                           G_ema if config['ema'] else None)
        state_dict['save_num'] = (
            state_dict['save_num'] + 1) % config['num_save_copies']

    # Use EMA G for samples or non-EMA?
    which_G = G_ema if config['ema'] and config['use_ema'] else G

    # Accumulate standing statistics?
    if config['accumulate_stats']:
        if not config['conditional']:
            y_.zero_()
        utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                        z_, y_, config['n_classes'],
                                        config['num_standing_accumulations'])

    # Save a random sample sheet with fixed z and y
    with torch.no_grad():
        if config['parallel']:
            fixed_Gz = nn.parallel.data_parallel(
                which_G, (fixed_z, which_G.shared(fixed_y)))
        else:
            fixed_Gz = which_G(fixed_z, which_G.shared(fixed_y))
    if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
        os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
    image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'],
                                                    experiment_name,
                                                    state_dict['itr'])
    torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                                 nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)
    # For now, every time we save, also save sample sheets
    utils.sample_sheet(which_G,
                       classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
                       num_classes=config['n_classes'],
                       samples_per_class=10, parallel=config['parallel'],
                       samples_root=config['samples_root'],
                       experiment_name=experiment_name,
                       folder_number=state_dict['itr'],
                       z_=z_)
    # Also save interp sheets
    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
        utils.interp_sheet(which_G,
                           num_per_sheet=16,
                           num_midpoints=8,
                           num_classes=config['n_classes'],
                           parallel=config['parallel'],
                           samples_root=config['samples_root'],
                           experiment_name=experiment_name,
                           folder_number=state_dict['itr'],
                           sheet_number=0,
                           fix_z=fix_z, fix_y=fix_y, device='cuda')


''' This function runs the inception metrics code, checks if the results
    are an improvement over the previous best (either in IS or FID, 
    user-specified), logs the results, and saves a best_ copy if it's an 
    improvement. '''

def test(G, D, G_ema, z_, y_, state_dict, config, sample, get_inception_metrics, 
         experiment_name, test_log):
    """
    Saving the appropriate metrics for sample quality (FID) and level of bias
    """
    print('Pre-loading pre-trained attribute classifier...')
    if config['n_classes'] == 2:
        clf_state_dict = torch.load(CLF_PATH)['state_dict']
    else:
        # multi-attribute
        raise NotImplementedError
    # load attribute classifier here
    clf = ResNet18(block=BasicBlock, layers=[2, 2, 2, 2], 
                    num_classes=config['n_classes'], grayscale=False) 
    clf.load_state_dict(clf_state_dict)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clf = clf.to(device)
    clf.eval()  # turn off batch norm

    # obtain classifier predictions for samples
    preds = classify_examples(clf, config)  # (10K,)
    fair_d = utils.fairness_discrepancy(preds, config['n_classes'])
    print('Fairness discrepancy metric is: {}'.format(fair_d))

    print('Gathering inception metrics...')
    if config['accumulate_stats']:
        utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                        z_, y_, config['n_classes'],
                                        config['num_standing_accumulations'])
    IS_mean, IS_std, FID = get_inception_metrics(sample,
                                                 config['num_inception_images'],
                                                 num_splits=10)
    print('Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' %
          (state_dict['itr'], IS_mean, IS_std, FID))
    # If improved over previous best metric, save approrpiate copy
    if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
            or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])
            or (config['which_best'] == 'fair' and fair_d < state_dict['best_fair_d'])
        ):
        print('%s improved over previous best, saving checkpoint...' %
              config['which_best'])
        utils.save_weights(G, D, state_dict, config['weights_root'],
                           experiment_name, 'best%d' % state_dict['save_best_num'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num'] = (
            state_dict['save_best_num'] + 1) % config['num_best_copies']
    state_dict['best_IS'] = max(state_dict['best_IS'], IS_mean)
    state_dict['best_FID'] = min(state_dict['best_FID'], FID)
    state_dict['best_fair_d'] = min(state_dict['best_fair_d'], fair_d)
    # Log results to file
    test_log.log(itr=int(state_dict['itr']), IS_mean=float(IS_mean),
                 IS_std=float(IS_std), FID=float(FID))

def update_FID(G, D, G_ema, state_dict, config, FID, experiment_name, test_log):
    print('Itr %d: PYTORCH UNOFFICIAL FID is %5.4f' %
          (state_dict['itr'], FID))
    # If improved over previous best metric, save approrpiate copy
    if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
            or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])):
        print('%s improved over previous best, saving checkpoint...' %
              config['which_best'])
        utils.save_weights(G, D, state_dict, config['weights_root'],
                           experiment_name, 'best%d' % state_dict['save_best_num'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num'] = (
            state_dict['save_best_num'] + 1) % config['num_best_copies']
    state_dict['best_FID'] = min(state_dict['best_FID'], FID)
    # Log results to file
    test_log.log(itr=int(state_dict['itr']), IS_mean=float(0),
                 IS_std=float(0), FID=float(FID))

def classify_examples(model, config):
    """
    classifies generated samples into appropriate classes 
    """
    import numpy as np
    model.eval()
    preds = []
    samples = np.load(config['sample_path'])['x']
    n_batches = samples.shape[0] // 1000

    with torch.no_grad():
        # generate 10K samples
        for i in range(n_batches):
            x = samples[i*1000:(i+1)*1000]
            samp = x / 255.  # renormalize to feed into classifier
            samp = torch.from_numpy(samp).to('cuda').float()

            # get classifier predictions
            logits, probas = model(samp)
            _, pred = torch.max(probas, 1)
            preds.append(pred)
        preds = torch.cat(preds).data.cpu().numpy()

    return preds
