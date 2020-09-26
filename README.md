# Demo_Matrix_Multiplication_backward

onvolution is the most widely used and important layer in deep learning neural networks for image classification or regression tasks. Its counterpart, transpose convolution or typically named as ConvTranspose, is also widely used in networks (e.g. U-Net[1], ResNet[2]) that requires re-sampling data back to the original image size so that they can be added to or concatenated with the original data to form skip layers. Due to the complexity involved in the forward and backward computation, both convolution and ConvTranspose do not seem as straightforward as other modules. Therefore, we plan to use the following 4 posts to explain the fundamentals of convolution and ConvTranspose with examples of custom implementations and hope to help clarify these concepts.

    Matrix multiplication and its custom implementation (this post)
    Conv2d and its custom implementation
    ConvTranpose2d and its custom implementation
    Application of Conv2d and ConvTranpose2d in Neural Networks

We take the 1st part to talk about matrix multiplication because it can be used in Convolution and ConvTranspose operations to make things simpler. In this post, we will focus on derivation of the gradients of matrix multiplication. While the derivation process may seem complex, the final results will be in a pretty simple form and are easy to remember. 

Please see the details at https://coolgpu.github.io/coolgpu_blog/github/pages/2020/09/22/matrixmultiplication.html.
