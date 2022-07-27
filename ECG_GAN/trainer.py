import os
import numpy as np
import sys

from torch import nn
import torch.nn.functional as F
import torch
from acgan_tools import Configure_Data, Animator
from Generator_model import Generator as Generator
from Discriminator_model import Discriminator
from acgan_dataloader import setup_seed, weights_init_normal
from Pictures_show import ECG_Show
from acgan_dataloader import sample_image, load_ECG_data

def save_all_imgs(animator, LEAD_NUM, generator, ecg_show, model= 'gen_4', type_gen= 'LSTM'):
    """ 
    LEAD_NUM: 0-11
    model: generator model name
    type_gen: generator type
    """
    animator.save_fig(model, LEAD_NUM + 1)
    fake_samples = sample_image(18, generator)
    ecg_show.plot_fake(fake_samples, LEAD_NUM + 1, model+'_'+type_gen)

def train(label_indices, LEAD_NUM = 0, model= 'gen_4', type_gen= 'LSTM'):
    """
    label_indices: training data.
    """
    # Set random seeds
    setup_seed(20)
    devices = [torch.device(f'cuda:{i}') for i in range(2)]
    
    # Const
    CLASSES = 9
    HIDDEN_DIM = 400
    EPOCH, BATCH_SIZE = 140, 32
    LR_G, LR_D =  0.0001, 0.0001
    L = 5500 # ECG signal length
    LAMBDA_R = 0.0125 # Reconstruction hyperparameters
    LAMBDA_E = 0.1 # Embedding hyperparameters

    # Loss functions
    adversarial_loss = torch.nn.BCELoss() # True sample and Fake sample
    auxiliary_loss = torch.nn.CrossEntropyLoss()
    l1_loss = nn.L1Loss()

    # Initialize generator and discriminator
    generator = Generator(CLASSES, HIDDEN_DIM, L)
    discriminator = Discriminator(L, CLASSES)

    generator_name = generator.__class__.__name__

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Change the device inplace
    generator.cuda()
    discriminator.cuda()
    generator = nn.DataParallel(generator, device_ids= devices)
    discriminator = nn.DataParallel(discriminator, device_ids= devices)

    # Configure data loader
    configure_data = Configure_Data(label_indices.dataset, label_indices.datalabel, BATCH_SIZE)
    dataloader = configure_data.output_dataloader()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr= LR_G, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr= LR_D, betas=(0.5, 0.999))

    # Animator
    animator = Animator(xlabel='epoch', ylabel= ['Loss', 'Accuracy', 'Loss'],xlim=[1, EPOCH], ylim= None,
                        legend=[['discriminator loss', 'generator loss'],
                                 ['discriminator acc_real', 'discriminator acc_fake'],
                                 ['ad_loss_fake_dis', 'ad_loss_fake_gen']]               
                        )
            
    # ECG real and fake plot
    ecg_show = ECG_Show(label_indices.dataset[:, LEAD_NUM,:]) # The chosen LEAD_NUM 0-11

    generator.train()
    discriminator.train()
    #--------Training--------#
    for epoch in range(EPOCH):
        for i, (imgs, labels) in enumerate(dataloader): # imgs: ECG signal(1350, 12, 5500)
            
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = torch.rand((batch_size, 1), dtype= torch.float, requires_grad= False).fill_(1.0).cuda()
            fake = torch.rand((batch_size, 1), dtype= torch.float, requires_grad= False).fill_(0.0).cuda()

            # Configure input
            real_imgs = imgs[:, LEAD_NUM, :].unsqueeze(1).cuda() # choose 1 lead in 12 leads (batch_size, 1, L)
            labels = labels.type(torch.long).cuda()
 
    # ----------------- Train Generator -----------------#
            
            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            gen_labels = torch.tensor(labels.cpu().numpy(), dtype= torch.long).cuda()

            if generator_name == 'Generator_3':
                z = torch.tensor(np.random.normal(0, 1, (batch_size, L)), dtype= torch.float).cuda()
                gen_imgs, gen_real, embed_imgs, embed_real = \
                generator(real_imgs ,z, gen_labels) # gen_imgs shape (batch_size, 1, L)
                validity, pred_label = discriminator(gen_imgs)

                reconstruct_loss = LAMBDA_R * (l1_loss(gen_imgs, gen_real))
                embed_loss = LAMBDA_E * (l1_loss(embed_imgs, embed_real))
                g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels)) \
                    + reconstruct_loss \
                    + embed_loss

            elif generator_name == 'Generator_4':
                z = torch.tensor(np.random.normal(0, 1, (batch_size, HIDDEN_DIM)), dtype= torch.float).cuda()
                gen_imgs = generator(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                validity, pred_label = discriminator(gen_imgs)
                reconstruct_loss = LAMBDA_R * (l1_loss(gen_imgs, real_imgs))
                g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels)) \
                    + reconstruct_loss
            
            elif generator_name == 'Generator':
                z = torch.tensor(np.random.normal(0, 1, (batch_size, HIDDEN_DIM)), dtype= torch.float).cuda()
                gen_imgs = generator(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                validity, pred_label = discriminator(gen_imgs)
                reconstruct_loss = LAMBDA_R * (l1_loss(gen_imgs, real_imgs))
                g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels)) \
                    + reconstruct_loss

    
            g_loss.backward()
            optimizer_G.step()            
    # --------------------- Train Discriminator ---------------------#

            optimizer_D.zero_grad()
            # Loss for real images
            real_pred, real_aux = discriminator(real_imgs)
            d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

            # Loss for fake images
            fake_pred, fake_aux = discriminator(gen_imgs.detach())
            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            # Calculate discriminator accuracy
            # pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            pred_real, pred_fake = real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()
            # gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
            
            gt_real, gt_fake = labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()

            # d_acc = np.mean(np.argmax(pred, axis=1) == gt)
            d_acc_real = np.mean(np.argmax(pred_real, axis=1) == gt_real)
            d_acc_fake = np.mean(np.argmax(pred_fake, axis=1) == gt_fake)

            d_loss.backward()
            optimizer_D.step()

            if i == len(dataloader) // batch_size:
                if generator_name == 'Generator_3':
                    animator.add(epoch + 1, (d_loss.item(), (g_loss-reconstruct_loss-embed_loss).item(), d_acc_real, d_acc_fake,
                                    adversarial_loss(fake_pred, fake).item(),
                                    adversarial_loss(validity, valid).item()
                                    ))
                elif generator_name == 'Generator_4':
                    animator.add(epoch + 1, (d_loss.item(), (g_loss-reconstruct_loss).item(), d_acc_real, d_acc_fake,
                                    adversarial_loss(fake_pred, fake).item(),
                                    adversarial_loss(validity, valid).item()
                                    ))
                elif generator_name == 'Generator':
                    animator.add(epoch + 1, (d_loss.item(), (g_loss-reconstruct_loss).item(), d_acc_real, d_acc_fake,
                                    adversarial_loss(fake_pred, fake).item(),
                                    adversarial_loss(validity, valid).item()
                                    ))

            if (i+1) % 10 == 0:
                if generator_name == 'Generator_3':
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc_real: %.4f%%, acc_fake: %.4f%%] [G loss: %.4f,\
                        Reconstruct loss: %.4f, Embed loss: %.4f]"
                        %(epoch+1, EPOCH, i+1, len(dataloader), 
                        d_loss.item(), 100 * d_acc_real, 100 * d_acc_fake, 
                        (g_loss-reconstruct_loss-embed_loss).item(), reconstruct_loss.item()/LAMBDA_R, 
                        embed_loss.item())
                    )
                elif generator_name == 'Generator_4':
                    if LAMBDA_R != 0:
                        print(
                            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc_real: %.4f%%, acc_fake: %.4f%%] [G loss: %.4f,\
                            Reconstruct loss: %.4f]"
                            % (epoch+1, EPOCH, i+1, len(dataloader), 
                            d_loss.item(), 100 * d_acc_real, 100 * d_acc_fake, 
                            (g_loss-reconstruct_loss).item(), 
                            reconstruct_loss.item()/LAMBDA_R)
                        )
            
                elif generator_name == 'Generator':
                     if LAMBDA_R != 0:
                        print(
                            "[Epoch %d/%d] [Batch %d/%d] [D loss: %.5f, acc_real: %.4f%%, acc_fake: %.4f%%] [G loss: %.6f,\
                            Reconstruct loss: %.4f]"
                            % (epoch+1, EPOCH, i+1, len(dataloader), 
                            d_loss.item(), 100 * d_acc_real, 100 * d_acc_fake, 
                            (g_loss-reconstruct_loss).item(), 
                            reconstruct_loss.item()/LAMBDA_R)
                        )
                
                
    save_all_imgs(animator, LEAD_NUM, generator, ecg_show, model, type_gen)

    generator.eval()
    with torch.no_grad():
        net_params = generator.state_dict()
    
    return net_params

def save_generator_params(lead_num, cur_name, net_params, cur_path):

    filename = f'{lead_num}lead_model_weights_{cur_name+1}.pth'
    torch.save(net_params, os.path.join(cur_path, filename))


if __name__ == '__main__':

    file_loc = '/home/alien/XUEYu/paper_code/Parameters/2018_Generator'
    model_names = ['Generator', 'Generator_rmv_tanh']
    
    cur_path = os.path.join(file_loc, model_names[-1])
    if not os.path.exists(cur_path):
        os.makedirs(cur_path)
    
    label_indices = load_ECG_data() # Load ecg data
    
    for lead_num in range(12):
        cur_name = len(os.listdir(cur_path))
        net_params = train(label_indices, lead_num, 'gen_1', 'Conv')
        save_generator_params(lead_num, cur_name, net_params, cur_path)