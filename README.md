# TSNE-CUDA
### A CUDA implementation of the t-SNE clustering algorithm

A quick, simple application for quickly creating t-SNE representations of data on NVIDIA GPUs. Meant for use by researchers on Penn State's ACI-ICS high performance compute cluster.

#### Download and Use

After downloading to an enviroment with an NVIDIA GPU, make sure you have nvcc findable in your path and make the executable. Running the application requires 3 arguments: the path to the csv file from the current executable, the amount of data points, and the amount of variables in each point (not including label). For example,

```
./tsne data/mnist_test 1000 784
```

#### Files Included
```
TSNE-CUDA
|-> data
|---> fashion_mnist-test.csv 
|---> mnist_test.csv
|-> tsne (executable)
|-> tsne.cu (source code)
|-> graph.py (produce graphs using matplotlib and generated output)
```

#### Speed
![alt text][tests]

This project was done with a partner for CMPSC 450, High Performance Computing
[tests]: /tests.jpg "Tests"
