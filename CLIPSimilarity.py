from PIL import Image
import torch
import pandas as pd
import requests
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util
from typing import List
import numpy as np


class CLIPSimilarity:  # implementation class for CLIP similarithy

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    @staticmethod
    def similarityScore(array: pd.Series, cutoff: int, model=None, processor=None):#returns 2n concatenated pandas frame of caption and sim score
        # TODO add memoization/pooling, use a more efficient datatype
        image = array[0]
        text = list(array[1])
        print(text, 'text')
        if model == None:  # set params
            model = SentenceTransformer(
                'sentence-transformers/clip-ViT-B-32-multilingual-v1')

        imageModel = SentenceTransformer('clip-ViT-B-32')

        if processor == None:
            processor = CLIPSimilarity.processor

        # inputs = processor(text=text, images=image,
        #                    return_tensors="pt", padding=True)
        textEmbed = model.encode(text)
        imageEmbed = imageModel.encode(image)
        # this is the image-text similarity score
        logits_per_image = util.cos_sim(textEmbed, imageEmbed).flatten()
        # we can take the softmax to get the label probabilities
        probs = torch.tensor(logits_per_image)
        probs = np.array(probs)
        # probs = np.array(torch.softmax(probs, 0))
        out = (probs, text,image)
        # out.sort(reverse=True)

        # filteredOut = np.array(out[:cutoff]).T

        # TODO FUCKING DISGUSTING CODE FIX IT 
        # dont care lmao
        return out


if __name__ == '__main__':  # testing code please ignore #TODO fix this to be up to date
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    image = Image.open(requests.get(url, stream=True).raw)
    text = [[image, "a photo of two cats"], #TODO Broken
            [image, "a photo of two dogs"]]
    
    df = pd.DataFrame([text])
    print(df.head())
    newdf = df.apply(CLIPSimilarity.similarityScore, args=(5,), axis=1)
    
    print(newdf.head())
