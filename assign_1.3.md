Convolution : Its a process of merging 2 sets of input and provide a new set. 
              In CNN, convolution is performed on the input data with the use
              of a feature extractor to extract the edges,gradients then textures
              , parts of object and then object.
              convolution is executed by sliding the filter over the input.
              At every location, a matrix multiplication is performed and 
              sums the result onto the feature map.
              
Filters/Kernels : Kernel is a feature extractor which on convolving with input 
              provides feature map.
              
Epochs :  Number of times data will be trained on the dataset.

1x1 Convolution : It looks only in one pixel of image mainly used to
          increase/decrease the number of channels as in below e.g..
          (32x32x10) * (1x1x10)x4 ===> 32x32x4 
          
3x3 Convolution : 3x3 convolution is mainly done to extract a feature where we 
            dot multiplication of matrix (kernel and input).


Feature Maps: The feature map is the output of kernel applied to the previous layer.

Activation Function: Activation function is used to get output of the layer for given set of 
                        input and which can be used as a input in the next layer

Receptive Field: It is the number of pixel seen by the kernel at a time. These are 2 type:
          Local receptive field , where the number of pixel seen in immediate previous layer 
          is counted. 
          Global receptive filed : number of layer the last layer sees the input.
          
       



