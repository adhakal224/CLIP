import os
import numpy as np
from PIL import Image
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from torchvision import datasets 
from sklearn.metrics import accuracy_score
from tqdm import tqdm

pwd = '/home/a.dhakal/active/user_a.dhakal/CLIP'
data_dir = pwd+'/data'
results_dir = pwd+'/results'
cifar_test = datasets.CIFAR10(root=data_dir, train=False, download=True)
classes = cifar_test.classes
cifar_targets = cifar_test.targets
cifar_data = cifar_test.data

clip_class = ['photo of a '+thing for thing in classes]
clip_data = [Image.fromarray(data) for data in cifar_data]
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") 

start = 0
end = 1000
final_probs = []
final_classes = []

for i in tqdm(range(10)):
  curr_data = clip_data[start:end]
  inputs = processor(text=clip_class, images=curr_data, return_tensors="pt", padding=True)
  outputs = model(**inputs)
  logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
  probs = logits_per_image.softmax(dim=1) 
  output_prob = probs.detach().numpy()
  output_class = output_prob.argmax(axis=1)
  final_probs.append(output_prob)
  final_classes.append(output_class)
  start = start+1000
  end = end+1000

final_classes = np.array(final_classes).reshape(-1,)
acc_map = final_classes == cifar_targets
clip_accuracy = np.mean(acc_map)
print(f'The accuracy is {clip_accuracy}')
np.save(results_dir+'/clip_preds.npy',final_classes)
np.save(results_dir+'/gt.npy',cifar_targets)












