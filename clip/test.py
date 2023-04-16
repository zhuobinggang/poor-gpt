from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from PIL import Image
import torch


processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32")
texts = ["a man", "a dog", "a cat", 'a lady', 'a taku', 'a loser', 'a successful people']
inputs = processor(text=texts, images=Image.open("taku.png"), return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

for text, prob in zip(texts, probs[0]):
    print(f'{text}: {round(prob.item() * 100, 2)}%')

