# Conditional-GAN-based-Image-Restoration-from-Speckle-Interference-Pattern

Project Idea: 
While imaging through a turbulent atmosphere, the images get speckled. 
This speckle can be modelled by adding a random phase to its amplitude and then a common Gaussian blurring. 
But if we get a speckled image, then at some points we get zero and know nothing about the information of that pixel. 
So, to increase our chances, we will take multiple shots (here, 3). 
So for a dataset, we first create a synthetic dataset as a pair (speckled image (3 channels), clean image (1 channel)). 
Then, using this dataset, we'll train a pix2pix GAN network and will see if we can restore the images or not. 
For the 3 channels, we implement a new random phase to model the dynamic condition of the atmosphere.


improvements
1. Instead of Gaussian impulse response use angular spectrum propagation for in which wavelength satellite imagery happens.
2. Calculate the optimal gamma for gamma transformation using this method. Find the gamma for which clean and speckled images have best SSIM.
3. Similarly find the optimal number of shots. for n=1 SSIM(speckled[0], clean), for n=2 mean(SSIM(speckled[0], clean), SSIM(speckled[1], clean))



questions:
1. How to get good result for this unstable training process for pix2pix GAN?