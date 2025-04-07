# Quarter Laplacian Filter

This repository implements Quarter Laplacian Filter for Edge Aware Image Processing using the approach mentioned in this
[paper](https://github.com/bonaventuredossou/dvip_project/blob/main/data/ICIP%201%20-%20QUARTER%20LAPLACIAN%20FILTER%20FOR%20EDGE%20AWARE%20IMAGE%20PROCESSING.pdf).

## Overview
Image smoothing is the fundamental operation in image processing. We use it to remove image details or noise in the
image. While performing smoothing operation, it is needed to remove small gradients and preserve the large ones which is
called edge preserving. This repository uses quarter window of Discrete Laplacian operator originally obtained from
discrete diffusion equation.

```math
U^{t+1}(x_i, y_j) = U^t(x_i, y_j) + c \Delta U^t(x_i, y_j)
```

### Implementation
It is implemented using classical box filter method but instead of using entire Laplacian Kernel this method uses
quarter window of Laplacian Kernel.

Standard Laplacian Kernels are as follows:

<pre>
[0,   1/4, 0  ]   [-1/16, 5/16, -1/16]   [1/12, 1/6, 1/12]
[1/4, -1,  1/4] , [5/16,  -1,    5/16] , [1/6,  -1,   1/6]
[0,   1/4, 0  ]   [-1/16, 5/16, -1/16]   [1/12, 1/6, 1/12]
</pre>

This implementation uses right most Kernel which is the most isotropic kernel as mentioned in the paper. Therefore, the 
quarter Laplacian filters would be as follows:

<pre>
     [1/3, 1/3, 0]        [0, 1/3, 1/3]        [0,   0,   0]        [0,     0, 0]
K1 = [1/3, -1,  0] , K2 = [0,  -1, 1/3]   K3 = [0,  -1, 1/3] , K4 = [1/3,  -1, 0]
     [0,   0,   0]        [0,   0,   0]        [0, 1/3, 1/3]        [1/3, 1/3, 0]
</pre>

We would obtain four feature maps from these kernels. 

```math
d_i = k_i * U, \forall i = 1,...,4
```

where * is convolution operator. Then only one feature map $d_{m(x,y)}(x, y)$ is selected, where

```math
m(x,y) = argmin\{|d_i(x,y)|; i = 1,...,4\}
```

### Installation

NOTE: Linux based OS or MacOS required to run this shell script.

1.	Clone the repository from git.
2.	Open the command line interface (CLI) of your OS and move to the project folder <i>ICIP1_quarter_laplacian/</i> 
      using cd command.
3.	Run the installation script using following command
	
	$ ./install.sh
          
this script will automatically create a new directory namely <i>env_dir/</i> to create a virtual environment, activate 
that environment and install all the required library into that.

NOTE: If you get permission denied error while running this script, kindly run the command
	
    $ chmod 755 install.sh

4.  Once the project is installed and environment is activated, <i>main.py</i> can be run using Python3 as follows 


    $ python3 main.py [image_name]

It is important to note that we are using Python3 for the implementation purpose. Also, user can provide image name with
its path as an optional parameter to <i>main.py</i> which will be considered as base original image and all operations 
will be performed on that image. If parameter is not provided, all smoothing operations will be performed on this 
[standard camera-man image](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.camera) which is 
similar to the one shown in paper of ICIP1.

### Authors
Chirag Vaghela
