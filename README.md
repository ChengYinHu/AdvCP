# AdvCP
Code and model for "Adversarial Color Projection: A Projector-based Physical-World Attack to DNNs" (Image and Vision computing 2023)
<p align='center'>
  <img src='test.png' width='700'/>
</p>

## Introduction
Recent advances have shown that deep neural networks (DNNs) are susceptible to adversarial perturbations. Therefore, it is necessary to evaluate the robustness of advanced DNNs using adversarial attacks. However , traditional physical attacks that use stickers as perturbations are more vulnerable than recent light-based physical attacks. 

In this work, we propose a projector-based physical attack called adversarial color projection (AdvCP), which performs an adversarial attack by manipulating the physical parameters of the projected light.
## Requirements
* python == 3.8
* torch == 1.8.0

## Basic Usage
```sh
python PSO_test.py --model resnet50 --dataset your_dataset
```

