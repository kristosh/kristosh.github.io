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

One of the most convenient things in the Neural Networks is that we do have a full control of all the parameters of our model. We know all the exact inner mechanisms that lead from the input to the output. In the case of the `GPT2` model we know all the operations: `positional-embeddings`, `positional-encoding`, `self-attention`, `feed-forward mechanisms` etc. We can intervene and make edits to parts of our model and investigate how these changes influence the output of the model.

Accordingly, being able to do this is a pretty core operation, and this is one of the main things TransformerLens supports! The key feature here is hook points. Every activation inside the transformer is surrounded by a hook point, which allows us to edit or intervene on it.

We do this by adding a hook function to that activation. The hook function maps current_activation_value, hook_point to new_activation_value. As the model is run, it computes that activation as normal, and then the hook function is applied to compute a replacement, and that is substituted in for the activation. The hook function can be an arbitrary Python function, so long as it returns a tensor of the correct shape


As a basic example, let's ablate head 7 in layer 0 on the text above.

We define a head_ablation_hook function. This takes the value tensor for attention layer 0, and sets the component with head_index==7 to zero and returns it (Note - we return by convention, but since we're editing the activation in-place, we don't strictly need to).

We then use the run_with_hooks helper function to run the model and temporarily add in the hook for just this run. We enter in the hook as a tuple of the activation name (also the hook point name - found with utils.get_act_name) and the hook function.

```python
layer_to_ablate = 0
head_index_to_ablate = 8

# We define a head ablation hook
# The type annotations are NOT necessary, they're just a useful guide to the reader
# 
def head_ablation_hook(
    value: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: HookPoint
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    print(f"Shape of the value tensor: {value.shape}")
    value[:, :, head_index_to_ablate, :] = 0.
    return value

original_loss = model(gpt2_tokens, return_type="loss")
ablated_loss = model.run_with_hooks(
    gpt2_tokens, 
    return_type="loss", 
    fwd_hooks=[(
        utils.get_act_name("v", layer_to_ablate), 
        head_ablation_hook
        )]
    )
print(f"Original Loss: {original_loss.item():.3f}")
print(f"Ablated Loss: {ablated_loss.item():.3f}")
```

Of course we can decide to ablate different head or different layer and we can see different divergence between the original and the ablated loss.

### A more involved example with running hooks

Now, we want to check a bit more involved task. In this example we will try to make a prompt that has a logical and expected answer to it and then make a corrupted equivalent prompt and monitor the difference in where the attention. Firstly, we will try to identify the difference in the loss between the original and the corrupted prompt:


```python
clean_prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"
corrupted_prompt = "After John and Mary went to the store, John gave a bottle of milk to"

clean_tokens = model.to_tokens(clean_prompt)
corrupted_tokens = model.to_tokens(corrupted_prompt)

def logits_to_logit_diff(logits, correct_answer=" John", incorrect_answer="Mary"):
    # model.to_single_token maps a string value of a single token to the token index for that token
    # If the string is not a single token, it raises an error.
    correct_index = model.to_single_token(correct_answer)
    incorrect_index = model.to_single_token(incorrect_answer)
    return logits[0, -1, correct_index] - logits[0, -1, incorrect_index]

# We run on the clean prompt with the cache so we store activations to patch in later.
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
clean_logit_diff = logits_to_logit_diff(clean_logits)
print(f"Clean logit difference: {clean_logit_diff.item():.3f}")

# We don't need to cache on the corrupted prompt.
corrupted_logits = model(corrupted_tokens)
corrupted_logit_diff = logits_to_logit_diff(corrupted_logits)
print(f"Corrupted logit difference: {corrupted_logit_diff.item():.3f}")
```


```python
# We define a residual stream patching hook
# We choose to act on the residual stream at the start of the layer, so we call it resid_pre
# The type annotations are a guide to the reader and are not necessary
def residual_stream_patching_hook(
    resid_pre: Float[torch.Tensor, "batch pos d_model"],
    hook: HookPoint,
    position: int
) -> Float[torch.Tensor, "batch pos d_model"]:
    # Each HookPoint has a name attribute giving the name of the hook.
    clean_resid_pre = clean_cache[hook.name]
    resid_pre[:, position, :] = clean_resid_pre[:, position, :]
    return resid_pre

# We make a tensor to store the results for each patching run. We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
num_positions = len(clean_tokens[0])
ioi_patching_result = torch.zeros((model.cfg.n_layers, num_positions), device=model.cfg.device)

for layer in tqdm.tqdm(range(model.cfg.n_layers)):
    for position in range(num_positions):
        # Use functools.partial to create a temporary hook function with the position fixed
        temp_hook_fn = partial(residual_stream_patching_hook, position=position)
        # Run the model with the patching hook
        patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[
            (utils.get_act_name("resid_pre", layer), temp_hook_fn)
        ])
        # Calculate the logit difference
        patched_logit_diff = logits_to_logit_diff(patched_logits).detach()
        # Store the result, normalizing by the clean and corrupted logit difference so it's between 0 and 1 (ish)
        ioi_patching_result[layer, position] = (patched_logit_diff - corrupted_logit_diff)/(clean_logit_diff - corrupted_logit_diff)
```


```python
# Add the index to the end of the label, because plotly doesn't like duplicate labels
token_labels = [f"{token}_{index}" for index, token in enumerate(model.to_str_tokens(clean_tokens))]
imshow(ioi_patching_result, x=token_labels, xaxis="Position", yaxis="Layer", title="Normalized Logit Difference After Patching Residual Stream on the IOI Task")
```