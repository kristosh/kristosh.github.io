---
layout: distill
title: How to explain the behavior of vision transformers?
description: This page's goal is to present techniques that can shed light on how Vision Transformers' models (ViTs) operate. We will first have a refresher on the ViTs and how they work. Then, we will introduce various methods to visualize the way that they function. These methods range from visualizing the attention maps to visualizing the queries keys and values, but also using the gradient similar to gradCAM approaches. We will make use of PyTorch implementation to demonstrate these methods. At the end of the blog post, there is a simple exercise that you will need to solve to portray some understanding of the way that ViT and the interpretability methods operate.

date: 2023-05-13
htmlwidgets: true

# anonymize when submitting
authors:
  - name: Anonymous

# do not fill this in until your post is accepted and you're publishing your camera-ready post!
authors:
  - name: Christos Athanasiadis
    url: https://www.linkedin.com/in/christos-athanasiadis-a3b51035/
    affiliations:
      name: UvA, Interpretability and Explainability in AI

# must be the exact same name as your blogpost
# bibliography: 2023-05-13-distill-example.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Vision Transformers (ViTs) Introduction
  - name: Methodology
  - name: Explainable VIts
  - name: TOSUBMIT
  - name: Conclusions
  - name: References
---

# Introduction


{% include figure.html path="assets/img/2023-06-23-Explainable-ViT/ViT_architecture.PNG" class="img-fluid" %}

What is a Visual Transformer? What is going on with the inner mechanism of it? How do they even work? Can we poke at them and dissect them into pieces to understand them better? These are some questions that we will try to answer in this post. Firstly, we will try to remind our reader about what is exactly a visual transformer and how it works. Secondly, we will try to revise methods that aim to shed light on their inner mechanisms. Our aim goal is to investigate all the current standard practices for visualizing the mechanisms that cause the ViT classifier to predict a specific class each time. These visualizations could be useful for:

- **the developer**: What’s going on inside when we run the Transformer on this image? Being able to look at intermediate activation layers. In computer vision - these are usually images! These are kind of interpretable since you can display the different channel activations as 2D images.

-  **the developer**: What did it learn? Being able to investigate what kind of patterns (if any) did the model learn. Usually, this is in the form of the question "What input image maximizes the response from this activation?", and you can use variants of “Activation Maximization” for that.

- **both the developer and the user**: What did it see in this image? Being able to Answer <em>What part of the image is responsible for the network prediction</em>, is sometimes called <em>Pixel Attribution</em>.

This tutorial was based on the code from the following tutorial: [https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html).

What we will cover in this tutorial are the following things:

- Firstly, we will create a simple ViT classifier trained in three classes. We will check how well it works and we will perform transfer-learning to improve its performance.
- Then, we will check several methods to shed light on how the ViT classifier works. More specifically, we will check two methods called: *Rollout methods** and **Gradient-based methods**

Finally, a simple TODO exercise will be provided to gauge the performance of different XAI methods for our built ViT model.

# Vision Transformers

The vanilla Transformer architecture was introduced by Vaswani et al. in 2017 [1], to handle sequential data and most specifically for natural language processing (NLP) for the machine translation task. Given the success of the Transformer architecture in NLP, Dosovitskiy et al. [2] proposed the Vision Transformer (ViT) architecture for image classification tasks. The ViT architecture is a pure transformer architecture that can handle images as input. Thus, unlike the vanilla Transformer, where the input was a sequence of text tokens, an image is a 3D grid of pixels. For that reason, similarly to the vanilla transformers, we will need to convert the 3D grid of pixels into a sequence of tokens. This could be done by splitting the image into non-overlapping patches. Each patch is then flattened into a 1D vector and then linearly projected into a vector of the same dimension as the token embeddings. Finally, these token embeddings are fed into the ViTs architecture in a similar way as the vanilla transformers. The basic blocks of the ViTs architecture can be seen in the previous image and are:

- **inputs**
- **linear projection (embedding layer)**
- **positional encoding**
- **transformer encoder**
    - **Layer normalization**
    - **multi-head self-attention**
    - **feed-forward neural network**
- **classification token**
- **MLP head**  
- **output predictions**

To help the reader comprehend all the above, we will provide a simple grouping of definitions and use the following two terms:

- **layers**: are basic elements that are used to build the architecture. For example, the multi-head self-attention can be considered as a layer. It's important to mention that a Transformer is composed usually of several Encoders. Each of these Enconders can be referred to in the literature as a layer as well.
- **blocks**: a grouping of layers (for instance the whole encoder can be seen as a block of layers), can be seen in Figure 1.

The ViT architecture is comprised of several stages:

- **Patch + Position Embedding (inputs)**: Turns the input image into a sequence of image patches and adds a position number to specify in what order the patch comes in.
- **Linear projection of flattened patches (Embedded Patches)**: The image patches get turned into an embedding, the benefit of using an embedding rather than just the image values is that an embedding is a learnable representation (typically in the form of a vector) of the image that can improve with training.
- **Norm**: This is short for <em>Layer Normalization</em> or <em>LayerNorm</em>, a technique for regularizing (reducing overfitting) a neural network, you can use LayerNorm via the PyTorch layer torch.nn.LayerNorm().
- **Multi-Head Attention**: This is a Multi-Headed Self-Attention layer or "MSA" for short. You can create an MSA layer via the PyTorch layer  <mark style="background-color:LavenderBlush;">torch.nn.MultiheadAttention()</mark>.
- **MLP (or Multilayer perceptron)**: An MLP can often refer to any collection of feedforward layers (or in PyTorch's case, a collection of layers with a forward() method). In the ViT Paper, the authors refer to the MLP as "MLP block" and it contains two <mark style="background-color:LavenderBlush;">torch.nn.Linear()</mark> layers with a  <mark style="background-color:LavenderBlush;">torch.nn.GELU()</mark> non-linearity activation in between them (section 3.1) and a torch.nn.Dropout() layer after each.
- **Transformer Encoder**: The Transformer Encoder, is a collection of the layers listed above. There are two skip connections inside the Transformer encoder (the "+" symbols) meaning the layer's inputs are fed directly to immediate layers as well as subsequent layers. The overall ViT architecture is comprised of a number of Transformer encoders stacked on top of eachother.
-  **MLP Head**: This is the output layer of the architecture, it converts the learned features of an input to a class output. Since we're working on image classification, you could also call this the "classifier head". The structure of the MLP Head is similar to the MLP block.


Breaking the hyperparameters down:

- **Layers**: How many Transformer Encoder blocks are there? (each of these will contain an MSA block and the MLP block)
- **Hidden size  D**: This is the embedding dimension throughout the architecture, this will be the size of the vector that our image gets turned into when it gets patched and embedded. Generally, the larger the embedding dimension, the more information can be captured, which leads to better results. However, a larger embedding comes at the cost of more compute. One niche issue here is that we need to distinguish the <em>layers</em> as hyperparameters that refer to the number of Transformer Encoder blocks and the <em>layer</em> as a building block of the encoder.
- **MLP size**: What are the number of hidden units in the MLP layers?
- **Heads**: How many heads are there in the Multi-Head Attention layers?
- **Params**: What is the total number of parameters of the model? Generally, more parameters lead to better performance but at the cost of more compute. You'll notice even ViT-Base has far more parameters than any other model we've used so far.

# Vision Transformers (ViTs) Tokenization

The standard Transformer receives as input a 1D sequence of token embeddings. To handle 2D images, we reshape the image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ into a sequence of flattened patches with size $\mathbf{x}_P \in \mathbb{R}^{N \times CP^2}$, when $H, W$ represent the height and the width of an image while $C$ represents the number of channels, then, $N$ is the number of patches and $P$ is the patch dimensionality. The Transformer uses constant latent vector size  $D$ through all of its layers, so we flatten the patches and map to $D$ dimensions with a trainable linear projection (Eq. 1). We refer to the output of this projection as the patch embeddings. If the input image is of size $224 \times 224 \times 3$ and the patch size is $16$ then the output should be of size $196 \times 768$, where the first dimension is the number of patches and the second dimension is the size of the patch embeddings $16\cdot 16\cdot 3 = 768$.

Having created the patches, it is time to create an implementation for the attention block which is the core unit of the ViTs architecture. The attention block is the same as the one in the vanilla transformers. It consists of a multi-head self-attention mechanism followed by a feed-forward neural network. The multi-head self-attention mechanism is used to capture the dependencies between the patches. The feed-forward neural network is used to capture the non-linear interactions between the patches. The following image portrays the mechanism of the attention block.

Here we will need to decide whether the input to our model will be the full image or the image patches. To decide that we will take into account that a lot of pre-trained models have as input the full image. Thus, for now, we will use the full image as input to our model. 

{% include figure.html path="assets/img/2023-06-23-Explainable-ViT/ViT_architecture.PNG" class="img-fluid" %}

# Load the pizza-sushi-steak dataset 🍕🍣🥩

In this section, we will implement a simple ViT classifier for the pizza-sushi-steak dataset. The dataset contains train and test folders with 450 images for training and 150 images for testing. We will start by providing some code for setting up our data. Firstly, we should download the dataset:

```python
image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    destination="pizza_steak_sushi")
```

Set up the paths for the training and testing data:

```python
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# Setup directory paths to train and test images
train_dir = image_path / "train"
test_dir = image_path / "test"
```

Then, we would like to perform some basic transformations to our data:

```python
# Create image size (from Table 3 in the ViT paper)
IMG_SIZE = 224

# Create transform pipeline manually
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
```

Then you will need to create the dataloaders for train and test sets:

```python
# Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False, # don't need to shuffle test data
      num_workers=num_workers,
      pin_memory=True,
  )
```

And return a batch of images and labels with the following code:

```python
image_batch, label_batch = next(iter(train_dataloader))
```
# Create a ViT classifier

Having loaded the data, now it's time to create a simple ViT classifier and fit our data. We will call the ViT model and pass the image batch to it. The output of the model will be the logits for each class. We will then use the cross-entropy loss to calculate the loss and the Adam optimizer to update the weights of the model. The following code will help you to create a simple ViT classifier:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seeds()
# Create an instance of ViT with the number of classes we're working with (pizza, steak, sushi)
vit = ViT(num_classes=len(cls_names))

# Setup the optimizer to optimize our ViT model parameters using hyperparameters from the ViT paper
optimizer = torch.optim.Adam(params=vit.parameters(),
    lr=3e-3, # Base LR from Table 3 for ViT-* ImageNet-1k
    betas=(0.9, 0.999), # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
    weight_decay=0.3) # from the ViT paper section 4.1 (Training & Fine-tuning) and Table 3 for ViT-* ImageNet-1k

# Setup the loss function for multi-class classification
loss_fn = torch.nn.CrossEntropyLoss()
# Train the model and save the training results to a dictionary
results = train_function(model=vit,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=10,
    device=device)
```

## Building the ViT model 

## Training function for the ViT model

## Measuring the performance of the ViT model

# Explainable Vision Transformers

# TODO

# Conclusions

In this tutorial, we have analyzed LIME a posthoc XAI technique. An explanation of how this technique works but also step-by-step the code to implement it. We have also seen how we can use LIME to explain image classifiers but also how to identify the bias in a classifier. 

# References
[[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. Gomez, {. Kaiser, and I. Polosukhin. Advances in Neural Information Processing Systems, page 5998--6008. (2017).](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf){: style="font-size: smaller"}

[[2] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, N, Houlsby, An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, International Conference on Learning Representations (2021).](https://openreview.net/pdf?id=YicbFdNTTy){: style="font-size: smaller"}

