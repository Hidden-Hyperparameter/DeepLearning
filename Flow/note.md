# Flow

## Small Bugs

1. Loss is `nan`: should use `Tanh` in last layer of `alpha` net.
2. Logdet: logdet should be `alpha*(1-mask)`, instead of just `alpha`.

## Large Bugs
1. Without `BatchNorm`, the log prob is 
2. `pre_process`: make it better, not necessary? Without it, log prob indeed negative!
2. batch size

## Log Prob is a very large positive number, but DON'T be afraid!

## How to implement `BatchNorm`? (Multiple Options)

1. No Pre-process, if batchnorm eval use running mean, then eval loss is really large (order of $10^{12}$).

(老师原来的网络就可以，现在这个就不行，我也不知道为什么！)

2. Question: what happen when generation(you don't have access to the statics of `x`?)