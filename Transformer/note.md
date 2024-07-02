# Transformer

## Bugs

1. Mask (how to add it to attention scores; how to construct the mask)
2. `nn.Transformer`: the output is irrelevant to the input. Why?

## Some Issues
- Batch size matters: at least 10 for a stable training.
    - Remove outliers (very long sentences) in dataset by clamping them to max length 100.