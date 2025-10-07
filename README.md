# pixelart-gen
A pixelart generator using ML. Currently WIP.

## Recommendations for Further Improvement
If you want to continue improving your VAE model:
- Try different architectures: ResNet blocks or skip connections can help preserve fine details.
- Post-processing: For pixel art specifically, you could apply quantization or nearest-neighbor upscaling to maintain the pixel art aesthetic.

TODO:
- fix pixelart dataset to dedupe train/valid splits
- do run with 0 kl weight
- use mnist dataset
- log std deviation instead of log variance
- push kl weight higher until both interpolations look good and reconstruction looks good (balance both)
- try interpolation on higher layers

interpolation ideas:
- encode 2 images
- show activations of all layers by running for loop and creating a large image composing all layers
- try to visualize the intermediate activations
- post images to wandb of intermediate activations
- visualize the features

Q: how do i visualize the features convolutional layers?
