---
layout: distill
title: How to explain the behavior of vision transformers?
description: This page's goal is to present techniques that can shed light in Vision Transformers' models (ViTs). We will first have a refresher on the ViTs architecture and how they work and we will investigate methods that can help us understand the behavior of these models.

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

But what is going on inside Vision Transformers? How do they even work? Can we poke at them and dissect them into pieces to understand them better?

(useful for the developer) What’s going on inside when we run the Transformer on this image? Being able to look at intermediate activation layers. In computer vision - these are usually images! These are kind of interpretable since you can display the different channel activations as 2D images.

(useful for the developer) What did it learn? Being able to investigate what kind of patterns (if any) did the model learn. Usually, this is in the form of the question "What input image maximizes the response from this activation?", and you can use variants of “Activation Maximization” for that.

(useful for both the developer and the user) What did it see in this image? Being able to Answer <em>What part of the image is responsible for the network prediction</em>, is sometimes called <em>Pixel Attribution</em>.

This page is based on the explanation and the code from the Deep Learning 1 course from the master program of Amsterdam University (UvA). The tutorial and the code can be found at the following link: [https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html).

The expectation from this tutorial 

# Vision Transformers

The vanilla Transformer architecture was introduced by Vaswani et al. in 2017 [1], to handle sequential data and most specifically for natural language processing (NLP) for the machine translation task. Given the success of the Transformer architecture in NLP, Dosovitskiy et al. [2] proposed the Vision Transformer (ViT) architecture for image classification tasks. The ViT architecture is a pure transformer architecture that can handle images as input. Thus, unlike the vanilla Transformer, where the input was a sequence of text tokens, an image is a 3D grid of pixels. For that reason, similarly to the vanilla transformers, we will need to convert the 3D grid of pixels into a sequence of tokens. This could be done by splitting the image into non-overlapping patches. Each patch is then flattened into a 1D vector and then linearly projected into a vector of the same dimension as the token embeddings. Finally, these token embeddings are fed into the ViTs architecture in a similar way as the vanilla transformers.

## Vision Transformers (ViTs) Tokenization

As mentioned above, the first step in ViTs is to convert the 3D grid to patches. To do that, we will make use 


```python
def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x

```

Having created the patches, it is time to create an implementation for the attention block which is the core unit of the ViTs architecture. The attention block is the same as the one in the vanilla transformers. It consists of a multi-head self-attention mechanism followed by a feed-forward neural network. The multi-head self-attention mechanism is used to capture the dependencies between the patches. The feed-forward neural network is used to capture the non-linear interactions between the patches. The following image portrays the mechanism of the attention block.

{% include figure.html path="assets/img/2023-06-23-Explainable-ViT/ViT_architecture.PNG" class="img-fluid" %}


The attention block is implemented as follows:

```python
class AttentionBlock(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, 
                                          dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x

```

Now we have all modules ready to build our own Vision Transformer. Besides the Transformer encoder, we need the following modules:
- A linear projection layer that maps the input patches to a feature vector of larger size. It is implemented by a simple linear layer that takes each 
 patch independently as input.
- A classification token is added to the input sequence. We will use the output feature vector of the classification token (CLS token in short) for determining the classification prediction.
- Learnable positional encodings that are added to the tokens before being processed by the Transformer. Those are needed to learn position-dependent information and convert the set to a sequence. Since we usually work with a fixed resolution, we can learn the positional encodings instead of having the pattern of sine and cosine functions.
- An MLP head that takes the output feature vector of the CLS token, and maps it to a classification prediction. This is usually implemented by a small feed-forward network or even a single linear layer.

With those components in mind, let’s implement the full Vision Transformer below:

```python

    class VisionTransformer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+num_patches,embed_dim))


    def forward(self, x):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out


# Conclusions

In this tutorial, we have analyzed LIME a posthoc XAI technique. An explanation of how this technique works but also step-by-step the code to implement it. We have also seen how we can use LIME to explain image classifiers but also how to identify the bias in a classifier. 

# References
[[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. Gomez, {. Kaiser, and I. Polosukhin. Advances in Neural Information Processing Systems, page 5998--6008. (2017).](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf){: style="font-size: smaller"}

[[2] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, N, Houlsby, An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, International Conference on Learning Representations (2021).](https://openreview.net/pdf?id=YicbFdNTTy){: style="font-size: smaller"}

