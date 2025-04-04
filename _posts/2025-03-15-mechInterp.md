---
layout: post
title: Mechanistic interpretability (to be filled soon)
date: 2025-03-15 16:40:16
description: A post on mechanistic interpretability and tranformerLens. This post is giving a basic introduction to mechanistic interpretability and ways to understand the inner mechanisms of transformer-style models like `GPT-2`. We will try to locate responsible attention maps for specific biases in several prompts and validate the causal connection by removing parts of the network. We will make use of different datasets to validate the behavior and visualize the results. Finally we will talk about circuits and attention heads.
tags: xAI, Interpretability
categories: sample-posts
---

 post on mechanistic interpretability and tranformerLens. This post is giving a basic introduction to mechanistic interpretability and ways to understand the inner mechanisms of transformer-style models like `GPT-2`. We will try to locate responsible attention maps for specific biases in several prompts and validate the causal connection by removing parts of the network. We will make use of different datasets to validate the behavior and visualize the results. Finally we will talk about circuits and attention heads.

#### Content of the post

- Introduction
- Loading models 
- tranformerLens
- Circuits

#### Check List

- [x] Brush Teeth
- [ ] Put on socks
  - [x] Put on left sock
  - [ ] Put on right sock
- [x] Go to school

Firstly, we would load several packages and libraries that will be really useful for us during the whole tutorial.

```python
import circuitsvis as cv
# Testing that the library works
cv.examples.hello("Neel")

# Import stuff
import torch
import torch.nn as nn
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import plotly.express as px

from jaxtyping import Float
from functools import partial

# import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix

torch.set_grad_enabled(False)
```


We can make use of the `HookedTransformer` functionality to load our model. In this tutorial we will nake use of `GPT-2`.  

```python
device = utils.get_device()
# NBVAL_IGNORE_OUTPUT
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
```


```python
model_description_text = """## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. See my explainer for documentation of all supported models, and this table for hyper-parameters and the name used to load them. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. 

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"""
loss = model(model_description_text, return_type="loss")
print("Model loss:", loss)

gpt2_text = "Whats is the co-capital of Greece according to the country's public opinion?"
gpt2_tokens = model.to_tokens(gpt2_text)
print(gpt2_tokens.device)
gpt2_logits, gpt2_cache = model.run_with_cache(gpt2_tokens, remove_batch_dim=True)

print(type(gpt2_cache))
attention_pattern = gpt2_cache["pattern", 0, "attn"]
print(attention_pattern.shape)
gpt2_str_tokens = model.to_str_tokens(gpt2_text)

print("Layer 0 Head Attention Patterns:")

cv.attention.attention_patterns(tokens=gpt2_str_tokens, attention=attention_pattern)

attn_hook_name = "blocks.0.attn.hook_pattern"
attn_layer = 0
_, gpt2_attn_cache = model.run_with_cache(gpt2_tokens, remove_batch_dim=True, stop_at_layer=attn_layer + 1, names_filter=[attn_hook_name])
gpt2_attn = gpt2_attn_cache[attn_hook_name]
assert torch.equal(gpt2_attn, attention_pattern)
```