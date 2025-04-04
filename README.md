# pixelart-gen
A pixelart generator using ML. Currently WIP.

TODO:
- [ ] Models wrapped in classes (lighning modules, take pytorch-lightning tutorial)
- [ ] Make variational by outputting a mean and a standard deviation
- [ ] Measure KL loss

"Variational" Autoencoder
VAE predict a mean and a standard deviation (sample from normal distribution)

Variational means sampling from a distribution.

- play with kl weight, if 0, then it's just a regular autoencoder (verify this behavior)
- log the mean and the standard deviation of the latent space to understand how much the kl weight affects it

## Recommendations for Further Improvement
If you want to continue improving your VAE model:
- Adjust the beta parameter: In your loss function, you have a beta parameter (currently set to 1.0) that controls the weight of the KL divergence term. Reducing this value (e.g., to 0.1 or 0.01) will make your model focus more on reconstruction quality at the cost of a less regular latent space.
- Increase model capacity: More convolutional layers or a larger latent space dimension can help capture more details.
- Try different architectures: ResNet blocks or skip connections can help preserve fine details.
- Post-processing: For pixel art specifically, you could apply quantization or nearest-neighbor upscaling to maintain the pixel art aesthetic.

You now have a command-line option to choose between deterministic and stochastic reconstruction, so you can easily compare the results and choose the approach that works best for your application.
