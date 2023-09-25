# Abstract
resolution image from a single low-resolution image. Due to the resolution limitation of large-scale imaging equipment, super-resolution technology enjoys wide range of real-world applications, especially in aerospace and medical fields. One bi g challenge is how to let the machine know that the output images are highresolution ones. Here, the key idea is to transform the unsupervised task into a supervised one by generatinginput low-resolution images through a pre-process of downsizing and upsizing high-resolution images, and the high-resolution ones are taken as the ground truth images. In this paper, we implement two supervised CNN
structures to solve the super-resolution problem, which are the super-resolution convolutional neural network
(SRCNN) and the residual dense network (RDN). To improve the model training efficiency, we optimize the
SRCNN model with batch normalization. Simulation results show that our SRCNN, SRCNN+BN and RDN
models all improve the quality of output images. Among them, SRCNN has the best PSNR and SSIM
performance, but RDN has great potential to achieve greater performance by increasing the number of
training epoches and the size of training set. Our codes are attached here.
