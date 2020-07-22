# CIFAR10_WideResNet9
WideResNet9 model on the CIFAR10 image dataset. Model trained for 30 epochs on a GPU achieves 92% test accuracy.  
- Data Processing steps: data normalization (mean=0, sd=1), data augmentation (4px padding, random 32x32 cropping, p=0.5 random horizontal flip).  
- Network architecture: using residual connections, batch normalization, ReLU activation.  
- To accelerate learning: learning rate scheduling, gradient clipping.  
- To prevent overfitting: weight decay (~l2 regularization).  
- Optimizer: Adam.  

