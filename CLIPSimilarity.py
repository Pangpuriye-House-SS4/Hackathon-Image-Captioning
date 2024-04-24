from PIL import Image
import pandas as pd
import requests
from transformers import CLIPProcessor, CLIPModel
from typing import List
import numpy as np

class CLIPSimilarity:  # implementation class for CLIP similarithy

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    @staticmethod
    # takes a PIL image and a list of text and outputs the softmax similarity for each entry of text
    def similarityScore(array:pd.Series,cutoff:int, model=None, processor=None):
        #TODO add memoization/pooling, use a more efficient datatype
        image = array[0]
        text = list(array[1:])
        if model == None:  # set params
            model = CLIPSimilarity.model

        if processor == None:
            processor = CLIPSimilarity.processor

        inputs = processor(text=text, images=image,
                           return_tensors="pt", padding=True)
        outputs = model(**inputs)

        # this is the image-text similarity score
        logits_per_image = outputs.logits_per_image
        # we can take the softmax to get the label probabilities
        probs = logits_per_image.softmax(dim=1).detach().numpy()[0]
        out = [(probs[i], text[i]) for i in (range(len(text)))]
        out.sort(reverse=True)
        
        filteredOut = np.array(out[:cutoff]).T
        
        return  pd.concat([pd.Series(filteredOut[0]),pd.Series(filteredOut[1])])#TODO FUCKING DISGUSTING CODE FIX IT

    

if __name__=='__main__':#testing code please ignore
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    text = ["a photo of two cats", "cats on a couch","a cat on a couch","two cats","cats and remotes on a couch"]
    print(CLIPSimilarity.SimilarityScore(image, text))
    image.show()
