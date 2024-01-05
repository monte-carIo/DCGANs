# DCGANs
- GANs is basically a framework where we simultaneously train two models including a generator (used to simulate the data distribution) and a discriminator (used to distinguish a sample from the data distribution or the generator). This can be considered as a minimax game, where we train D (discriminator) to maximize the probability of assigning the correct label to both training examples and samples from G (generator) and we simultaneously train G to minimize $log(1-D(G(z)))$ (z is a random noise):

  $ \min _G \max _D V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))] $

- DCGANs have the basics like GANs but there are some additional features, they use 2 main layers are convolutional layer and the transposed convolutional layer.
- Our goal is to find how to implement this in code with pytorch and CIFAR10 data set.
