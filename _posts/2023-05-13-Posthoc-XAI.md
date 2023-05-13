---
layout: distill
title: Posthoc explainability in AI
description: This page's main focus is to analyze a branch of explainable & interpretable AI (XAI) called posthoc XAI. We will analyze theory, taxonomy, applications, shortcomings of posthoc XAI approaches and apply them on image classification using popular CNN architectures and explain their black box nature. Part of the assessemnet for this tutorial/workshop, will be some research questions that needs be answered by you. These questions can be found all over this blogspot using the <mark>TOSUBMIT</mark> tag and will be summarized them at the end of the blogspot.
  
date: 2023-05-13
htmlwidgets: true

## anonymize when submitting 
authors:
  - name: Christos Athanasiadis
    url: "https://www.linkedin.com/in/christos-athanasiadis-a3b51035/"
    affiliations:
      name: UvA, Interpretability and Explainability in AI

# do not fill this in until your post is accepted and you're publishing your camera-ready post!
# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton 

# must be the exact same name as your blogpost
bibliography: 2023-05-13-Posthoc-XAI.bib  

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

## Introduction

This workshop's core focus is to analyze a branch of explainable & interpretable AI (XAI) called posthoc XAI. We will analyze theory, taxonomy, applications, shortcomings of posthoc XAI approaches and apply them on image classification using popular CNN architectures and explain their <em>black box</em> nature. Part of the assessemnet for this tutorial/workshop, will be some research questions that needs be answered by you. These questions can be found all over this notebook and will be summarized them at the end.

The learning objectives (ILOs) for this tutorial can be listed as follows:
- <em>Learning basic terminology for XAI and introduced to one possible taxonomy</em>,
- <em>Getting familiar with several Posthoc techniques for XAI (by a thorough list of references)</em>,
- <em>Test & compare these approaches using CNN classifier</em>,
- <em>Investigate and discover potential shortcoming of these techniques</em>,
- <em>Discovering means for mitigating problems in XAI methods</em>.

<font color='red'><b>TOSUBMIT</b></font>
Please submit answers on the questions marked "ToSubmit" in one workshop-report, together with your answers to the ToSubmit questions in the Lab 1 notebook. Use a single pdf with max. of 2 pages. Please submit as required via canvas/assignments. Include your name + student number. This report will be graded.

<font color='red'><b>TODO</b></font>
The code below is ready to run for the largest part. In order to ensure that you do an effort to understand the code we do ask you to finalize a few parts of the code, usually just one or a few lines of code are necessary. You do not have to submit the code. 

## Introduction to XAI

A ill-famed shortcoming in Deep Learning models is their *black-box* or *opaque* nature. Reasoning on their behavior, decisions and predictions is an fundamental task when developers debugging their models but also when AI users assessing the level of *trustworthiness* that they have on these models. It is really fundamental, especially when one is planning to take actions using these systems, or when deploying AI in high-stake real-world applications (such as law, finance, tumor analysis etc.). Understanding of the inner <em>black-box</em> mechanisms also provides insights into the model, which can be used to transform an <em>untrustworthy</em> model or prediction into a <em>trustworthy</em> one. Finally, GDPR regulations require DL models to be transparent for reasons of fairness.

### Trust AI

There is a widespread research in what it means a user to trust AI models and how explainability & interpretability can help mitigating opaqueness in AI. For simplicity reasons, we will employ here a really an basic definition: <em>trusting an AI algorithm means to be in a shape to anticipate its outcome and verify its performance (contractual trust)</em>.The following paper provide some good analysis on this aspect [Formalizing Trust in Artificial Intelligence: Prerequisites, Causes and Goals of Human Trust in AI]. If users do not trust a model or a prediction, they will probably not use it. 

We can seperate the notation of trust into two parts:
- Trusting a prediction, i.e. whether a user trusts an individual prediction (the output of the model given a particular input) sufficiently to take some action based on it.
- Trusting a model, i.e. whether the user trusts a model to behave in reasonable ways if deployed (for the whole input space).

### Interpretability accuracy trade-off

With the increasing popularity of AI the introduced approaches methodologies became more efficient but at the same time, increased their complexity. Think as an example the simple rule-based approaches in taking decisions. While they offer a really easy way to be explained, however, oftently, they lack of good accuracy in real-world problems. As it is apparent by the figure, there is a tradeoff between the achieved accuracy and the interpretability of the well-known machine learning approaches.

{% include figure.html path="assets/img/2023-05-13-Posthoc-XAI/tradeoff.png" class="img-fluid" %}


### Explainability & interpretability

By explaining a *prediction*, we define the visual artifacts that provide qualitative understanding of the relationship between the instance's components (e.g. words in text, patches in an image) and the model's prediction. We argue that explaining predictions is an important aspect in getting humans to trust and use machine learning effectively, if the explanations are faithful and intelligible. *To increase the trust in predictions and models Explainable and Interpretable AI (XAI) came at the rescue*. 

At this point, I would like to adotp a way to distinguish the following two terms: *interpretability* and *explaibility* using Ajay Thampi’s Interpretable AI book (https://www.manning.com/books/interpretable-ai):

- *Interpretability*: It is the degree to which we can consistently estimate what a model will predict given an input, understand how the model came up with the prediction, understand how the prediction changes with changes in the input or algorithmic parameters and finally understand when the model has made a mistake. Interpretability is mostly discernible by experts who are either building, deploying or using the AI system and these techniques are building blocks that will help you get to explainability.
- *Explainability:* Goes beyond interpretability in that it helps us understand in a human-readable form how and why a model came up with a prediction. It explains the internal mechanics of the system in human terms with the intent to reach a much wider audience. Explainability requires interpretability as building blocks and also looks to other fields and areas such as Human-Computer Interaction (HCI), law and ethics.

### Mitigating algorithmic bias using XAI

 Interpretability is a useful debugging tool for detecting bias in machine learning models. It might happen that the machine learning model you have trained for automatic approval or rejection of credit applications discriminates against a minority that has been historically disenfranchised. Your main goal is to grant loans only to people who will eventually repay them. The incompleteness of the problem formulation in this case lies in the fact that you not only want to minimize loan defaults, but are also obliged not to discriminate on the basis of certain demographics. This is an additional constraint that is part of your problem formulation (granting loans in a low-risk and compliant way) that is not covered by the loss function the machine learning model was optimized for.

Explainability techniques could help identify whether the factors considered in a decision reflect bias and could enable more accountability than in human decision making, which typically cannot be subjected to such rigorous probing.

- Bias in online recruitment tools
- Bias in word associations
- Bias in online ads
- Bias in facial recognition technology
- Bias in criminal justice algorithms

### Taxonomy of XAI techniques 

In the following picture you can find a possible way to categorize XAI methods:

{% include figure.html path="assets/img/2023-05-13-Posthoc-XAI/XAI_categories.png" class="img-fluid" %}


Firstly, XAI methods can be categorized into model-based and posthoc approaches. The former is trying to render the model itself interpretable and thus explain its behavior in clear and easy manner. While the later category, after having generate a classificaiton model that behaves as a black box and its too complicate to be inherently explain, aims at explaning the bevavior of the model after the training procedure.

The posthoc methods could be also divided between global and local methods. The global methods trying to explain the model around a specifc input sample while the global techniques are trying to do that on a global scale.

Another way to distinguinsh XAI models is between Model-specific or model-agnostic? Model-specific interpretation tools are limited to specific model classes. The interpretation of regression weights in a linear model is a model-specific interpretation, since – by definition – the interpretation of intrinsically interpretable models is always model-specific. Tools that only work for the interpretation of e.g. neural networks are model-specific. Model-agnostic tools can be used on any machine learning model and are applied after the model has been trained (post hoc). These agnostic methods usually work by analyzing feature input and output pairs. By definition, these methods cannot have access to model internals such as weights or structural information.  

Local or global? Does the interpretation method explain an individual prediction or the entire model behavior? Or is the scope somewhere in between? Read more about the scope criterion in the next section.

In this tutorial we will focus in several posthoc explanation and its subcategories for vision. We can categorize posthoc methods in three basic categories:
- <em>Gradient based methods</em> Many methods compute the gradient of the prediction (or classification score) with respect to the input features. The gradient-based methods (of which there are many) mostly differ in how the gradient is computed. Since in these methods we need to calcualte the gradient of the models, thus, we need to have access to the models themselves, therefore, model-specific techniques.
- Surrogate methods.
- Perturbation-based methods.
- <em>Feature of pixel attribution methods</em> highlight the features or pixels that were relevant for a certain image classification by a neural network.

In this tutorial, tou will apply a few techniques for feature attribution: 
**Feature attribution**
- **Leave one out (LOO)**
- **LIME** method
- **XAI counterfactuals** 
- **SHAP**
**Gradient-based:** 
- **Saliency mapping**: use gradients to understand what image pixels are most important for classification
- **Integrated gradients** for the same purpose
- **GradCam** to understand what areas of the image are important for classification. 


## Gradient-based methods

[LIME tutorial](../../2023/gradient_based_feature_attribution/)

## LIME

[LIME tutorial](../../2023/LIME/)

## Conclusion

In this post, we have seen two ways of using language for RL. There have been a lot of other ways recently in this direction. Some examples of these are

- <d-cite key="lampinen-icml22a"></d-cite> augment policy networks with the auxiliary target of generating explanations and use this to learn the relational and causal structure of the world
- <d-cite key="kumar-neurips22a"></d-cite> use language to model compositional task distributions and induce human-centric priors into RL agents.

Given the growth of pre-trained language models, it is only a matter of time before we see many more innovative ideas come around in this field. Language, after all, is a powerful tool to incorporate structural biases into RL pipelines. Additionally, language opens up the possibility of easier interfaces between humans and RL agents, thus, allowing more human-in-the-loop methods to be applied to RL. Finally, the symbolic nature of natural language allows better interpretability in the learned policies, while potentially making them more explainable. Thus, I see this as a very promising direction of future research