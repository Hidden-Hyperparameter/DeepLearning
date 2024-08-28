# VQVAE

Please note that it is important to back-propogate the reconstruction loss to encoder by copying the gradients from the decoder to the encoder. It is mentioned in the paper.

Moreover, it is important to adjust the ratio between reconstrcution loss and the commitment loss. The reconstruction loss should be significantly more important than the commitment loss.