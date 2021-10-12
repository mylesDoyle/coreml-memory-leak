# Reproduction of memory leak for CoreML inference on iOS device

This repository contains a simple example to reproduce the memory leak observed when running inference on a CoreML model on an iOS device (specifically tested on the iPhone12). To reproduce the memory leak, you must first generate a model by converting a PyTorch model to a CoreML `MLModel`, and then build and launch the model in a mobile application using XCode.

This example assumes you have installed and loaded the following environment listed.

## System environment:
 - coremltools version:  `5.0b5`:
 - OS: build on MacOS targetting iOS for mobile application:
 - macOS version: Big Sur (version 11.4)
 - iOS version: 14.7.1 (run on iPhone 12)
 - XCode version: Version 12.5.1 (12E507)
 - How you install python: Install from source
 - python version: [3.8.10](https://www.python.org/ftp/python/3.8.10/)
 - How you install Pytorch: Install from source
 - PyTorch version: 1.8.1

## Generate CoreML `MLModel`

A simple model involving three convolution and one pixel-shuffle layers is used to reproduce the memory leak. This PyTorch model is converted to a CoreML `MLModel` so that it can be used in the iOS application.

The CoreML model is built by converting a simple PyTorch model 
`coreml-memory-leak/convert_torch_to_coreml.py` contains a simple PyTorch model and a function to convert the Torch model to a CoreML `MLModel`.
	- Simply run this script to produce a CoreML model expecting input tensor shape `[1,48,480,270]` and outputs a tensor of shape `[1,3,3840,2160]`

	```python
	python convert_torch_to_coreml.py
	```
	- This produces the model `toy.mlmodel`.

## Run mobile application

The iOS application `TestApp` can be used to launch the generated model on an iOS device. The application was taken and adapted from [PyTorch's source `TestApp` example](https://github.com/pytorch/pytorch/tree/master/ios). This application should be built and run using XCode. You may need to modify the input arguments and for the application, which can be set as follows:

- `input_type` : `float`
- `coreml_model` : `true`
- `input_dims` : `1,48,480,270`
- `iter` : `50`
- `warmup` : `1`

You will also need to move the `MLModel` to `coreml-memory-leak/TestApp/TestApp/model.pt`. 