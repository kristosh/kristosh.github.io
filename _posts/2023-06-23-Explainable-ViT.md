---
layout: distill
title: How to explain the behavior of vision transformers?
description: This page's goal is to present a PostHoc feature attribution XAI methodology called LIME (Local Interpretable Model-agnostic Explanations) and demonstrate how it can be used to explain image classification tasks. You will be guided through the code and the results of the LIME algorithm. Part of this workshop will be some research questions that need to be answered by you. These questions can be found in the last part of the tutorials and you will encounter these questions with the <mark>TOSUBMIT</mark> tag. You will need to submit your answers to these questions in the form of a .pdf file on Canvas. The deadline for submitting your answers is the .. of June 2023. 

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
  - name: Local Interpretable Model-agnostic Explanations
    subsections:
    - name: Interpretable Representations
    - name: LIME approach details
    - name: Code implementation
    - name: LIME explanation 
    - name: Creating Perturbations of image
  - name: Results
  - name: Adversarial attacks
  - name: TOSUBMIT
  - name: Conclusions
  - name: References
---

# Local Interpretable Model-agnostic Explanations


{% include figure.html path="assets/img/2023-06-23-Explainable-ViT/ViT_architecture.PNG" class="img-fluid" %}
