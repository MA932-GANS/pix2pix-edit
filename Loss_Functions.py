import numpy as np
import matplotlib.pyplot as plt
import os
    
data_dir = os.getcwd() + '/Losses'
os.chdir(data_dir)

D = np.loadtxt('discrim_loss.csv', delimiter=',')
G = np.loadtxt('gen_loss_GAN.csv', delimiter=',')
L1 = np.loadtxt('gen_loss_L1.csv', delimiter=',')
x = np.loadtxt('time.csv', delimiter=',')

# convert time into hours
t = x[::-1]/3600

plt.figure(figsize=(12,8))
plt.plot(t,D, label ='Discriminator Loss')
plt.plot(t,G, label='Generator Loss GAN')
# plt.plot(t,L1, label='Generator Loss L1')
plt.legend()
plt.xlabel('Time, hours')
plt.ylabel('Loss')

plt.savefig('Losses.png')