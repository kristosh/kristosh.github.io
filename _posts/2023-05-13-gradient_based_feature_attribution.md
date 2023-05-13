---
layout: distill
title: Gradient-based feature attribution for Vision
description: This page's main focus is to analyze a branch of explainable & interpretable AI (XAI) called posthoc XAI. We will analyze theory, taxonomy, applications, shortcomings of posthoc XAI approaches and apply them on image classification using popular CNN architectures and explain their black box nature. Part of the assessemnet for this tutorial/workshop, will be some research questions that needs be answered by you. These questions can be found all over this blogspot using the <mark>TOSUBMIT</mark> tag and will be summarized them at the end of the blogspot.
  
date: 2023-05-13
htmlwidgets: true

## anonymize when submitting 
authors:
  - name: Christos Athanasiadis
    url: "https://www.linkedin.com/in/christos-athanasiadis-a3b51035/"
    affiliations:
      name: UvA, Interpretability and Explainability in AI
  - name: Peter Heemskerk 
    url: ""
    affiliations:
      name: UvA, Interpretability and Explainability in AI


# must be the exact same name as your blogpost
bibliography: 2023-02-11-Posthoc-XAI.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Introduction to XAI
  - name: Gradient-based methods
  - name: Conclusion


# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---




# Gradient-based feature attribution for Vision

Before starting with the explanation of the gradient-based methodologies, we provide some useful code for all the necessarry packages loacing a pre-trained VGG model (in Imagenet) but also code to load a image for a local directory in PyTorch. You can access the code of this tutorial in the following [google colab link](https://colab.research.google.com/drive/1zWmtpOTXfxv1Hxwl7G5heU73vz90iNhl?usp=sharing).  

The packages that you will need to import are the following ones:
```python
# set-up environment
import numpy as np
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os  # necessary
import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow   # specific for colab
os.environ['KMP_DUPLICATE_LIB_OK']='True'
```

### Load a pretrained model

We will make use of the VGG19 CNN network and ImageNet. 

- ImageNet is a large collection of images. 
- VGG19 is a convolutional neural network architecture. 
- We can load a version that is trained on ImageNet and that can detect objects in 1000 classes. 

- Read about and understand VGG ConvNet and Imagenet for background. 

The first step is that using the pytorch library, we load the pretrained version of VGG19. 

Since we will not train the model we set the model in evaluation mode.

```python
# load model
# model_type = 'vgg19'
model = models.vgg19(pretrained=True)

# run it on a GPU if available:
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('cuda:', cuda, 'device:', device)
model = model.to(device)

# set model to evaluation
model.eval()
```

The output should look like this:

```python
Output exceeds the size limit. Open the full output data in a text editorVGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace=True)
    (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
...
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```

### Load and preprocess the images:

We have provided a few images of wildlife, but please also use you own imagery. Set the path to your data-file and load an image. 

VGG-19 works best if image is normalised. Image should also be in the correct tensor format. 

```python
def pre_processing(obs, cuda):
    # Students should transpose the image to the correct tensor format. 
    # Students should ensure that gradient for input is calculated       
    # set the GPU device
    if cuda:
        torch_device = torch.device('cuda:0')
    else:
        torch_device = torch.device('cpu')

    # normalise for ImageNet
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    obs = obs / 255
    obs = (obs - mean) / std

    # make tensor format that keeps track of gradient
    # BEGIN for students to do
    obs = np.transpose(obs, (2, 0, 1))       
    obs = np.expand_dims(obs, 0)
    obs = np.array(obs)
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=torch_device)
    # END for students to do
    return obs_tensor

```

```python
%matplotlib inline
#The line above is necesary to show Matplotlib's plots inside a Jupyter Notebook
from matplotlib import pyplot as plt

# read the image and convert it - Set your pathto the image
img = cv2.imread('elephant-zebra.png')
#img = cv2.imread(datafiles+ 'R.png')
#img = cv2.imread(datafiles+ 'elephant/Elephant2.jpeg')
# img = cv2.imread(datafiles+ 'shark/Shark1.jpeg')
if (type(img) is np.ndarray):
  img = cv2.resize(img, (224, 224))
  img = img.astype(np.float32)
  img = img[:, :, (2, 1, 0)]
  print('img:', img.shape)


else:
  print('image not found - set your path to the image')
```

```python

fig = plt.figure(frameon=False, facecolor='white')
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax = plt.imshow(img/255)
plt.show()
```

### Predict class

We can easily predict the class, and the softmax score of that prediction:

````python
def predict(input, model, target_label_idx, cuda):
    # Makes prediction after preprocessing image 
    # Note that output should be torch.tensor on cuda
    output = model(input)                        
    output = F.softmax(output, dim=1) # calc output from model 
    if target_label_idx is None:
      target_label_idx = torch.argmax(output, 1).item()
    index = np.ones((output.size()[0], 1)) * target_label_idx
    index = torch.tensor(index, dtype=torch.int64) 
    if cuda:
      index = index.cuda()                     # calc prediction
    output = output.gather(1, index)           # gather functionality of pytorch
    return target_label_idx, output 

# test preprocessing
# you can check that the VGG network gives a correct prediction. E.g. 385 and 386 are 'Indian Elephant'and 'African Elephant'
input = pre_processing(img, cuda)          # preprocess: image (normalise, transpose, make tensor on cuda, requires_grad=True)
output = predict(input, model, None, cuda)
print('output:', output)
````

**TODO 1**
- Upload at least two images to your directory with the first one containing a single animal and the second image with two animals.
- Run the classifier with both images.
- Look up the predicted categories and the ImageNet labels.
- Look up the indices corresponsing to each of the animals in your images.

The following snippets might be useful:

```python
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
print(categories[385])
categories.index('Indian elephant')
```

### Compute the gradient with respect to the input pixels

Now that we can predict the class of an object, we will try to understand what image pixels are most important for the prediction using <em>feature attribution methods</em>. The first technique that we will make use is the saliency maps. In short this approach determines the gradient of the output w.r.t to the input. 

The idea of Saliency maps (called <em> Vanilla Gradient </em> as well), introduced by Simonyan et al. (https://arxiv.org/pdf/1312.6034.pdf) as one of the first pixel attribution approaches. The core idea is really simple and what needs to be done is to calculate the gradient of the loss function for the class we are interested in with respect to the input features. This gives us a map of the size of the input features with negative to positive values.

The recipe for this approach is as follows:

- **Perform a forward pass** of the image ($\mathbf{x}_0$) of interest using the network $\mathcal{F}(\mathbf{x}_0)$.
- **Compute the gradient** of class score of interest with respect to the input image ($\mathbf{x}_0$): $g(\mathbf{x}_0) = \frac{\partial \mathcal{F}}{\partial \mathbf{x}_0} $.
- **Visualize the gradients**: You can either show the absolute values or highlight negative and positive contributions separately.

### The instructions for the PyTorch code:

We have set the model in eval mode, but we can still catch the gradients of the input-image if ask PyTorch to do this and then do some backward calculation. That is what you need to do. So complete the procedure in order that:
- Input should be preprocessed (and converted into a torch tensor).
- Set the <mark>required_gradient=True</mark> on the input tensor.
- Calculate the output (with previous method predict).
- Set the gradient to zero and do a backward on the output. 
- Gradients w.r.t input can now be found under input.grad

````python
def calculate_outputs_and_gradients(inputs, model, target_label_idx, cuda=False):
    # Calculates the gradient of the output w.r.t. the input image
    # The result should be a gradients numpy matrix of same dimensions as the inputs
    predict_idx = None
    gradients = []
    for input in inputs:                             # for every image
        input = pre_processing(input, cuda)  
        input.requires_grad=True
        print (target_label_idx)
        _, output = predict(input, model, target_label_idx, cuda)
        # clear grad
        model.zero_grad()
        ## BEGIN student code
        # Perform a backward pass on the output and collect the gradient w.r.t. the input
        # Store this gradient in the variable 'gradient' 
        output.backward()
        gradient = input.grad.detach().cpu().numpy()[0]  # do backward and gather gradients of input
        ## END student code
        gradients.append(gradient)
    gradients = np.array(gradients)
    return gradients, target_label_idx
````

```python	
# calculate the gradient and the label index
#gradients, label_index = calculate_outputs_and_gradients([img], model, None, cuda)    
gradients, label_index = calculate_outputs_and_gradients([img], model, target_label_idx=15, cuda=False)    
gradients = np.transpose(gradients[0], (1, 2, 0))

#Note that if target_label_idx == None, the calculate_output)and_gradients function assume:
#            target_label_idx = torch.argmax(output, 1).item()

# Please note that the dimensions of gradients are same as dimensions of input
print('gradients', gradients.shape) 
# Please note that gradients are positive and negative values
print(gradients[:3, :3, 0])
```

```python
print(label_index)
```


<font color='green'><b>ToDo 2</b></font>
For your image with two animals, consider both ID's as target_label_idx.
+ Run the classifier with each ID
+ After running the forward pass, compute the gradients

<font color='blue'><b>ToThink 2</b></font>
Are the gradients the same when you use different target classes? Why?

### Visualize the gradients

Try to visualise the image and the saliency map. 

**Tip:** take absolute values of the gradients and maximize over all three channels. 

```python

# Retrieve the saliency map and also pick the maximum value from channels on each pixel.
# In this case, we look at dim=2. Recall the shape of gradients (width, height, channel)

def plot_gradients(img, gradients, title):
  # plots image (dimensions: Width X Heigth X 3) and gradients (dimensions: Width X Heigh x 3) - both numpy arrays
  saliency = np.max(np.abs(gradients), axis=2)       # takes maximum over 3 color channels                                                 
  # Visualize the image and the saliency map
  fig, ax = plt.subplots(1, 2)
  ax[0].imshow(img/255)
  ax[0].axis('off')
  ax[1].imshow(saliency, cmap='hot')
  ax[1].axis('off')
  plt.tight_layout()
  fig.suptitle(title)
  plt.show()

plot_gradients(img, gradients, 'The Image and Its Saliency Map')

```

<font color='red'><b>ToSubmit 1</b></font>
In your report, include your two images, the two saliency map, the two target  label, the predicted label and the likelihood your models assigns to each label. Add a caption explaining very briefly (1 or 2 sentences) whether there's a difference and why.

### Issues with saliency maps and vanilla gradients (saturation)

Vanilla Gradient methods, notoriously, are facing saturation problems, as explained in Avanti et al. (2017). When the ReLU is used, and when the activation goes below zero, then, the activation is limited at zero and does not change any more. Hence, the activation is saturated. 

Therefore, multiple strategies have been proposed to deal with that issue. One of them is Gradient-weighted Class Activation Map (<em>Grad-Cam</em>) that instead of calculating the gradient to the input image it make use of the last convolutional layer.

### Gradient-weighted Class Activation Mapping (Grad-CAM)

# **4 Gradient-weighted Class Activation Mapping (Grad-Cam).**

Unlike saliency maps, in the **Grad-Cam** approach the gradient is not backpropagated all the way back to the image, but (usually) to the last convolutional layer in order to generate a visualization map that highlights important regions of the input.

A naive visualization approach could be the following:

- Simply employ the values for each feature map, (of the last convolutional layer), 
- Then, average these feature maps and overlay this over our image (by rescaling back to initial size).  

However, while simple, it is not really helpful approach, since these maps encode information for all classes, while we are interested in a specific class. **Grad-CAM** needs to figure out the importance for each of the $k$ feature map $A_k \in \mathbb{R}^{w \times h}$ ($w$ the width and $h$ the height of the features maps) in respect to our class $c$ of interest.

We have to weight each pixel of each feature map with the gradient before averaging over the feature maps $A_k$. This heatmap is send through the ReLU function which set all negative values to zero. The reason for that is that we are only interested in the parts that contribute to the selected class $c$ and not to other classes. The final feature map is rescaled back to the original image size. We then overlay it over the original image for producing the final visualization.

**Grad Cam recipe:**

- Forward-propagate the input image $\mathbf{x}_0$ through the convolutional VGG19 network by calculating the $\mathcal{F}(\mathbf{x}_0)$.
- Obtain the score for the class of interest, that means the activation before the softmax layer.
- All the rest classes' activations should be set to zero.
- Back-propagate the gradient of the class of interest to the last convolutional layer before the fully connected layers:

$$\frac{\partial y_{c}}{\partial A^k}$$

- Weight each feature map "pixel" by the gradient for the class. Indices $i$ and $j$ refer to the width and height dimensions:

$$\alpha^{c}_{k} = \overbrace{\frac{1}{Z} \sum_i \sum_j}^{\text{global averaging pooling}} \underbrace{\frac{\partial y_{c}}{\partial A^{k}_{ij}}}_{\text{gradients of the backpropagation}}$$

This means that the gradients are globally pooled.

- Calculate an average of the feature maps, weighted per pixel by backpropagated gradient.
- Apply ReLU to the averaged feature map.

$$  L_{ij}^c = ReLU \sum_k \alpha^{c}_{k} A^{k}_{ij}$$

We now have a heatmap $L^c$ for the class $c$.

- Regarding the visualization: Scale values of the $L^c$ to the interval between 0 and 1. Upscale the image and overlay it over the original image.

In our classification example this approach uses the activation map of the final convolutional layer (with VGG: the final features layer). Note that such an Activation Map can be a block of $14 \times 14 \times 512$, where the $14 \times 14$ indicated a grid on the image (noted by subscripts i and j) and the 512 is the number of channels (features, noted by the letter k). **Grad Cam** pools the Activation Map over the channels, and it gives a weight equal to the contribution of each channel to the prediction. This contribution of each channel is calculated by taking the gradient of the output w.r.t. the Activation Map and then pool this over the spacial ($14\times14$) dimensions. 

For the calculation of the gradient w.r.t the Activation Map we need a little PyTorch trick since this gradient cannot be accessed by default. The PyTorch trick is called a 'hook'. We can register a hook on a tensor of the network. With a hook we can define a little program that is executed when the tensor is touched during a backward pass. In our case we register a hook on the Activation Map we want to study and that is the 36th layer of the VGG19 convolutional "features" layer. The hook needs to be registered during a forward pass, so we will redefine the forward pass for our model. 

There is a nice youtube tutorial on pytorch and hooks https://www.youtube.com/watch?v=syLFCVYua6Q. (22 minutes but I think it is worth it) 

### Define a new VGG model including a hook

The VGG() class is based on the pretrained models.vgg19 that we know now.

In the init, the Activation Map we want to study is defined. That is the output of the first 36 feature layers.

In the activations_hook method we define our hook that will store the gradient calculated on the tensor in self.gradients. 

In the forward we execute all VGG layers 'by hand'. The hook is registered on the output of the first 36 feature layers. And then the remaining layers are defined.   

When defined, we load this model, move it to our GPU if available and put the model in eval mode. 

### Activity:
Finish the code below by finishing the method get_activations_gradient. 

```python
class VGG(nn.Module):
    # VGG class builds on vgg19 class. The extra feature is the registration of a hook, that
    # stores the gradient on the last convolutional vgg layer (vgg.features[:36] in self.gradient)

    def __init__(self, model):
        super(VGG, self).__init__()
        
        # get the pretrained VGG19 network
        self.vgg = models.vgg19(pretrained=True)
        # self.vgg = model

        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:36]
        
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        # get the classifier of the vgg19
        self.classifier = self.vgg.classifier
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations: it stores the calculated grad (on our tensor) in self.gradients.
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        # gives the output of the first 36 'feature' layers
        x = self.features_conv(x)
        
        # register the hook (note: h is a handle, giving the hook a identifier, we do not use it here)
        h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))

        # apply the remaining classifying
        x = self.classifier(x)

        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        ## Should return the gradients of the output with respect to the last convolutional layer
        ## BEGIN Students TODO
        return self.gradients
        ## END students TODO
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)    

vgg = VGG(model)
print('cuda:', cuda, 'device:', device)
vgg = vgg.to(device)
vgg.eval()
```

Now calculate the gradients of a prediction w.r.t. the activation map.**

For that we do a prediction with our newly defined model vgg, and perform a backward on the output (the logit of the prediction vector that is largest). After the backward, the gradients w.r.t the activation map are stored in self.gradient:

```python
# get the most likely prediction of the model

input = pre_processing(img, cuda)          # preprocess: image (normalise, transpose, make tensor on cuda, requires_grad=True)
print(input.shape)

pred, output = predict(input, vgg, None, cuda)                         
print(pred, output)                        

# also with our newly created VGG(), you should find a correct class (2=shark, 385/386 = elephants)
```
And finally:

```python
output.backward()
```

```python
# pull the gradients out of the model
gradients = vgg.get_activations_gradient()
print('gradients:', gradients.shape)

# pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
print('pooled gradients:', pooled_gradients.shape)

# get the activations of the last convolutional layer
activations = vgg.get_activations(input).detach()

# weight the channels by corresponding gradients
for i in range(512):
    activations[:, i, :, :] *= pooled_gradients[i]
    
# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
# heatmap = np.maximum(heatmap, 0)
heatmap = torch.maximum(heatmap, torch.tensor(0))

# normalize the heatmap
heatmap /= torch.max(heatmap)
# END students TODO

print('heatmap:', heatmap.shape)

# draw the heatmap
heatmap = heatmap.cpu().numpy().squeeze()
plt.matshow(heatmap)
```


<font color='green'><b>ToDo 3</b></font>
For your image with 2 animals and each of the two target categories:
+ Perform the forward pass again, but now with our adapted VGG model
+ Draw the Grad-CAM heatmaps $L^c$. 

Code snippet that might be useful:
```
gradients, label_index = calculate_outputs_and_gradients([img], vgg, target_label_idx=None, cuda=True)    
gradients = np.transpose(gradients[0], (1, 2, 0))
```

### Overlaying heatmaps and iamges:
Now we have the heatmap, we can overlay it on the original image. 

```python
# draw the image
print('img:', img.shape)
plt.matshow(img/255)
```

```python
print(img.shape, heatmap.shape)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

print(img.shape, img.min(), img.max())
print(heatmap.shape, heatmap.min(), heatmap.max())

super_img = heatmap * 0.4 + img * 0.6
super_img = np.uint8(super_img)
plt.matshow(super_img)
```

<font color='red'><b>ToSubmit 2</b></font>
In your report, include the two Grad-CAM heatmaps for the image with two animals. Add a caption explaining very briefly (1 or 2 sentences) whether there's a difference and why.

### Path-integration methods - Integrated Gradients (IG)

As a reminder, the problem that want to study in this tutorial is to find a way to attribute importance in the input features of the vector $\mathbf{x}_i \in \mathbb{R}^{D}$ given the result of the classification from a classifier $\mathcal{F}$.

Suppose that we have a funtion $\mathcal{F}: \mathbb{R}^{D} \to [0, 0, ... , 1, ... , 0, 0] \in \mathbb{R}^{M} $ which represent a neural network. The input to this network are data $\mathbf{X} = (\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n) \in \mathbb{R}^{N\times D}$ we would like to calculate a vector $\mathbf{\alpha}_0 = (\alpha_1, \alpha_2, ..., \alpha_D) \in \mathbb{R}^{D}$ which is the contribution of the input vector $\mathbf{x}_0$ to the prediction $\mathcal{F}(\mathbf{x}_i)$.

Path-attribution methods in contrast with the gradient methods that we have mentioned before (saliency maps and grad-cam) compare the current image  $\mathbf{x}$ to a reference image $\mathbf{x}^{\prime}$ which can be for instance a black image (or a white image or an image containing random noise). The difference in actual and baseline prediction is divided among the pixels.

### IG approach

As a reminder, the problem that want to study in this tutorial is to find a way to attribute importance in the input features of the vector $\mathbf{x}_i \in \mathbb{R}^{D}$ given the result of the classification from a classifier $\mathcal{F}$.


Suppose that we have a funtion $\mathcal{F}: \mathbb{R}^{D} \to [0, 0, ... , 1, ... , 0, 0] \in \mathbb{R}^{M} $ which represent a neural network. The input to this network are data $\mathbf{X} = (\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n) \in \mathbb{R}^{N\times D}$ we would like to calculate a vector $\mathbf{\alpha}_0 = (\alpha_1, \alpha_2, ..., \alpha_D) \in \mathbb{R}^{D}$ which is the contribution of the input vector $\mathbf{x}_0$ to the prediction $\mathcal{F}(\mathbf{x}_i)$.


Path-attribution methods in contrast with the gradient methods that we have mentioned before (saliency maps and grad-cam) compare the current image  $\mathbf{x}$ to a reference image $\mathbf{x}^{\prime}$ which can be for instance a black image (or a white image or an image containing random noise). The difference in actual and baseline prediction is divided among the pixels.

### Calculate the integrated gradients with PyTorch recipe.

Recipe for calculating the IG in our example: 
  - **Choose a baseline image**. You can make use of a black/white or an white noise image. 
  - **Build a series of inputs**, each input consist of the baseline plus an additional fraction of the input-image. The final input is the baseline plus the full image. Choose your fraction at 20. 
  - For each of these inputs, **calculate the gradients of the input** w.r.t. the prediction (using methods under 2 above). Take the average of all these gradients.
  - **Calculate the difference of image and baseline**: I-B. And calculate Integrated Gradient = (I-B)*average of gradients. 
  - If you have chosen for another baseline, e.g. for a uniform random generated baseline, then perform this procedure for multiple samples. 

#### Integrated Gradients with one baseline

```python
# integrated gradients
def integrated_gradients(inputs, model, target_label_idx, baseline, steps=50, cuda=False):
    # determine baseline
    if baseline is None:
        baseline = 0 * inputs 
    # scale inputs and compute gradients
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
    grads, _ = calculate_outputs_and_gradients(scaled_inputs, model, target_label_idx, cuda)

   # BEGIN students TODO
    avg_grads = np.average(grads[:-1], axis=0)      # why 51 steps and then remove final result ?
    avg_grads = np.transpose(avg_grads, (1, 2, 0))
    delta_X = (pre_processing(inputs, cuda) - pre_processing(baseline, cuda)).detach().squeeze(0).cpu().numpy()
    delta_X = np.transpose(delta_X, (1, 2, 0))
    integrated_grad = delta_X * avg_grads
    # END students TODO
    return integrated_grad
```

#### Integrated Gradients with a sample of random baselines

```python	
def random_baseline_integrated_gradients(inputs, model, target_label_idx, steps, num_random_trials, cuda=False):
    # when baseline randomly generated, take some samples and average result
    # BEGIN students TODO
    all_intgrads = []
    random_baseline = 255.0 * np.random.random(inputs.shape)
    for i in range(num_random_trials):
        integrated_grad = integrated_gradients(inputs, model, target_label_idx, baseline=random_baseline, steps=steps, cuda=cuda)
        all_intgrads.append(integrated_grad)
        print('the trial number is: {}'.format(i))
    avg_intgrads = np.average(np.array(all_intgrads), axis=0)
    # END students TODO
    return avg_intgrads
```

<font color='green'><b>ToDo 4</b></font>
Investigate how well integrated gradients can determine what parts of the image your models is looking at for different target categories. Also investigate whether the zero baseline or a sample of random baselines gives you clearer feature attributions.

Code snippets that might be useful:

```
# calculate the integrated gradients 
print('img:', img.shape, 'label_index', label_index)

# for zero baseline
int_gradients_zerobl = integrated_gradients(img, model, label_index, baseline=None, steps=50, cuda=cuda)
print('DONE')
# for random baselines, we average over number of trials
int_gradients_randombl = random_baseline_integrated_gradients(img, model, label_index, steps=50, num_random_trials=5, cuda=cuda)    
print('DONE')

# calculate saliency
gradients, _ = calculate_outputs_and_gradients([img], model, None, cuda) 
gradients = np.transpose(gradients[0], (1, 2, 0))

# combine it all in one image
plot_gradients(img, gradients, 'The Image and Its Saliency Map')
plot_gradients(img, int_gradients_zerobl, 'Image and Integrated Gradients with Zero Baseline')
plot_gradients(img, int_gradients_randombl, 'Image and Integrated Gradients with sample of Random Baselines')
```

<font color='red'><b>ToSubmit 3</b></font>

Include in your  Workshop-0 report two images  with results applying  integrated gradients illustrating the strongest differences you have found (i.e., manipulating target categories, baselines, number of samples, or number of steps along the integration path). Include a brief caption that describes the experiment and your interpretation.


## LIME

[LIME tutorial](../../2022/LIME/)

## Conclusion

In this post, we have seen two ways of using language for RL. There have been a lot of other ways recently in this direction. Some examples of these are

- <d-cite key="lampinen-icml22a"></d-cite> augment policy networks with the auxiliary target of generating explanations and use this to learn the relational and causal structure of the world
- <d-cite key="kumar-neurips22a"></d-cite> use language to model compositional task distributions and induce human-centric priors into RL agents.

Given the growth of pre-trained language models, it is only a matter of time before we see many more innovative ideas come around in this field. Language, after all, is a powerful tool to incorporate structural biases into RL pipelines. Additionally, language opens up the possibility of easier interfaces between humans and RL agents, thus, allowing more human-in-the-loop methods to be applied to RL. Finally, the symbolic nature of natural language allows better interpretability in the learned policies, while potentially making them more explainable. Thus, I see this as a very promising direction of future research