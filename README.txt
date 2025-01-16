To launch the code:

conda env create -f environment.yml
conda activate deepblur
python -m demo.py

You will need an internet connection and a valid conda installation, with GPU.

This is the code associated to the paper:
Debarnot, V., & Weiss, P. (2024). Deep-blur: Blind identification and deblurring with convolutional neural networks. Biological Imaging, 4, e13. doi:10.1017/S2633903X24000096

The published paper is available here:
https://www.cambridge.org/core/services/aop-cambridge-core/content/view/34430C58046C7ECC1D28B3683B84ED89/S2633903X24000096a.pdf/deepblur_blind_identification_and_deblurring_with_convolutional_neural_networks.pdf

It contains the architecture and weights of our trained models for 400x400 grayscale images. 
A simple demo is available to show how it works.

Troubleshooting: pierre.weiss@cnrs.fr
Note: we do not intend to release all the codes (e.g. training), because a new updated version is being developed under DeepInv https://deepinv.github.io/deepinv/.
