import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import PCA_approach.PCA as PCA
import torchvision.utils as vutils
from os import listdir


# Thinks to keep in mind: have a different learning rate for mapping network!
# Make outputvalues of mapping network between zero and one and make it possible to work with that, square or cubic of  the output of the mapping network!!
# Maybe first try to autoencode the images and from that on do transferlearning.
# Regularize that image should be between 0 and 1
# multiply EV with mean of eigenvalues for this eigenvector
# Implement loading old parameter saves for further training
# Dot multiply mean face, with eigenvectors. and get mean of eigenvalues
# normalize Noise, 0 1

workers = 1

# Batch size during training
batch_size = 16

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 50

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 300
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Number of Eigenvectors that get used
n_components = 120

class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)




def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



class MappingNet(nn.Module):
        def __init__(self, ngpu):
            super(MappingNet, self).__init__()
            self.ngpu = ngpu
            self.lin_1 = nn.Linear(50,45)
            self.lin_2 = nn.Linear(45,40)
            self.lin_3 = nn.Linear(40,35)
            self.lin_4 = nn.Linear(35,30)
            self.lin_out = nn.Linear(30,120)
            self.act = nn.LeakyReLU()#nn.Tanh()
            

        def forward(self, x):
           #x = x.view(1,1,-1)
            x = self.act(self.lin_1(x))
            x = self.act(self.lin_2(x))
            x = self.act(self.lin_3(x))
            x = self.act(self.lin_4(x))
            x = self.lin_out(x)

            return x




#################### Helper Methods   ##############################


#path=os.getcwd()+"/training_saves/"
def load_model(path):
    model = MappingNet(1)#CARD_NET()
    model.load_state_dict(torch.load(path))
    return model



def pca_init(n_components):
    emojis = PCA.load_data_all_color()
    mean_face = np.mean(emojis,axis=1)/255
    e_values, e_vectors,pca = PCA.eigenface(emojis, n_components)

    
    std_vec = PCA.eigenvalue_range(emojis,pca,n_components)
    
    for i in range(n_components):
        e_vectors[i,:] = std_vec[i] * e_vectors[i,:]

    return torch.from_numpy(e_vectors),mean_face,pca




def training(old_save=None):
      


    

    dataroot = os.getcwd() + "/../emoji_data/emojis_jpg_root"
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device)
    fixed_noise = torch.randn(64, nz, device=device)
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)


    # Initialize PCA
    e_vectors,mean_face,pca = pca_init(n_components)
    

    batched_mean_face = np.zeros((batch_size,mean_face.shape[0]))
    for i in range(batch_size):
        batched_mean_face[i,:] = mean_face.copy()



    netD = Discriminator(ngpu).to(device)
    netM = MappingNet(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)


    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    # Probably needs a different learning rate?
    optimizerM = optim.Adam(netM.parameters(), lr=0.1, betas=(beta1, 0.999))

    real_label = 1
    fake_label = 0

    D_losses = [0]
    G_losses = [0]
    img_list = []
    iters = 0
    D_batchlosses = [0]
    G_batchlosses = [0]

    disc_once = False
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        G_losses_temp = []
        D_losses_temp = []
        
        for i, data in enumerate(dataloader, 0):
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            
            noise = (torch.randn(b_size, nz, device=device)*2)-1
            
            
            # Here comes the PCA output at the moment its just noise
            a = netM(noise)
            eigenvalues = a.double() #torch.sign(a)*torch.pow(a,2).double()##############################################
            fake = torch.from_numpy(batched_mean_face[:b_size,:])

            for i in range(b_size):
                fake[i,:] += torch.matmul(eigenvalues[i,:],e_vectors)
            #print("Fake shape: " + str(fake.shape))
            fake = fake.float()
            #fake = torch.randn(b_size, 3, 64, 64, device=device)
            label.fill_(fake_label)
            
            # Classify all fake batch with D
            output = netD(fake.detach().view(b_size,3,64,64)).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            # Only update if D_losses higher than g_losses
            if((D_losses[-1]) * 3 >= G_losses[-1] or D_x < 0.7 or (epoch%5 == 0 and not(disc_once))):
                print("I AM HERE")
                disc_once = True
                optimizerD.step()


            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netM.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake.view(b_size,3,64,64)).view(-1)
            # Calculate G's loss based on this output
            #print(torch.mean(fake).detach().numpy())
            errG = criterion(output, label) + 0.001 * (torch.mean(torch.abs(fake)) - 0.5)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerM.step()

            # Output training stats
            if i % 15 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            G_losses_temp.append(errG.item())
            D_losses_temp.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    eigenvalues = netM(fixed_noise).detach().cpu().double()
                    fake = torch.from_numpy(np.zeros((64,12288)))
                    fake[:16,:] = torch.from_numpy(batched_mean_face.copy())
                    fake[16:32,:] = torch.from_numpy(batched_mean_face.copy())
                    fake[32:48,:] = torch.from_numpy(batched_mean_face.copy())
                    fake[48:64,:] = torch.from_numpy(batched_mean_face.copy())
                    
                    for i in range(64):
                        fake[i,:] += torch.matmul(eigenvalues[i,:],e_vectors)
                img_list.append(vutils.make_grid(fake.reshape(64,64,64,3), padding=2, normalize=True))

            iters += 1

        disc_once = False
        # Save parameters
        D_batchlosses.append(np.mean(D_losses_temp))
        G_batchlosses.append(np.mean(G_losses_temp))
        torch.save(netM.state_dict(),os.getcwd() + "/training_saves/"+ str(D_batchlosses[-1])[:10] + "_g" +str(G_batchlosses[-1])[:10] + ".pth")


    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.gcf()
    plt.show()



def show_some_pictures(name,n_components,nz):
    # Do PCA
    ev,mean_face,pca = pca_init(n_components)
    fixed_noise = torch.randn(64, nz)
    # Get Mapping Net
    net = load_model(os.getcwd() + "/training_saves/" + name + ".pth")
    net.eval()
    # Get Mean face
    batched_mean_face = np.zeros((16,mean_face.shape[0]))
    for i in range(16):
        batched_mean_face[i,:] = mean_face.copy()
    
    # Get eigenvalues from mapping network
    output = net(fixed_noise).double()
    # Init fakes with mean face
    print(output[0,:])
    fake = torch.from_numpy(np.zeros((64,12288)))
    fake[:16,:] = torch.from_numpy(batched_mean_face.copy())
    fake[16:32,:] = torch.from_numpy(batched_mean_face.copy())
    fake[32:48,:] = torch.from_numpy(batched_mean_face.copy())
    fake[48:64,:] = torch.from_numpy(batched_mean_face.copy())
    
    # Linear combination of Eigenvectors
    for i in range(64):
        fake[i,:] += torch.matmul(output[i,:],ev)

    # Show the fakes
    fake = fake.detach().numpy()
    plt.figure()
    for i in range(64):
        plt.subplot(8,8,i+1)
        unten = np.min(fake[i,:])
        oben = np.max(fake[i,:])
        print(unten)
        print(oben)
        print(np.std(fake[i,:]))
        #b = (fake[i,:].reshape(64,64,3)- np.mean(fake[i,:])+0.5)
        #print(b[b> 1].size)
        

        plt.imshow((fake[i,:].reshape(64,64,3)- np.mean(fake[i,:])+0.5))
    plt.show()



def gui_init(save_path, seed=0, n_components=120, nz=50, pca_init=None):
    if not pca_init:
        ev,mean_face,pca = pca_init(n_components)
    else:
        ev, mean_face, pca = pca_init

    if(seed):
        torch.manual_seed(seed)
    fixed_noise = torch.randn(64, nz)
    # Get Mapping Net
    net = load_model(save_path)
    net.eval()

    batched_mean_face = np.zeros((16,mean_face.shape[0]))
    for i in range(16):
        batched_mean_face[i,:] = mean_face.copy()

    output = net(fixed_noise).double()
    fake = torch.from_numpy(np.zeros((64,12288)))
    fake[:16,:] = torch.from_numpy(batched_mean_face.copy())
    fake[16:32,:] = torch.from_numpy(batched_mean_face.copy())
    fake[32:48,:] = torch.from_numpy(batched_mean_face.copy())
    fake[48:64,:] = torch.from_numpy(batched_mean_face.copy())

    # Linear combination of Eigenvectors
    for i in range(64):
        fake[i,:] += torch.matmul(output[i,:],ev)
    fake = fake.detach().numpy()


    list_of_images = []
    for i in range(64):
        unten = np.min(fake[i,:])
        oben = np.max(fake[i,:])
        #print(unten)
        #print(oben)
        #print(np.std(fake[i,:]))
        #b = (fake[i,:].reshape(64,64,3)- np.mean(fake[i,:])+0.5)
        #print(b[b> 1].size)
        
        list_of_images.append((fake[i,:].reshape(64,64,3) - unten)/(oben-unten))
    return list_of_images



if __name__ == '__main__':   

    #show_some_pictures("0.00422660_g5.78255901",120,50)
    training()