# WaveNet

## Loss doesn't change

Issue: the receptive field is wrong. The casuality is also not correctly implemented.

Reason: the final reshape is wrong.