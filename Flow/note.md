# Flow

## Bugs

1. Loss is `nan`: should use `Tanh` in last layer of `alpha` net.
2. Logdet: logdet should be `alpha*(1-mask)`, instead of just `alpha`.
3. Acc issue: should use `torch.float64` (not sure???? God)
4. Log prob is positive not necessarily false!