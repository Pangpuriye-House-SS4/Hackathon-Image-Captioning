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
    # # takes a PIL image and a list of text and outputs the softmax similarity for each entry of text
    # def similarityScore(array:pd.Series,cutoff:int, model=None, processor=None):
    #     #TODO add memoization/pooling, use a more efficient datatype
    #     image = array[0]
    #     text = list(array[1:])
    #     if model == None:  # set params
    #         model = CLIPSimilarity.model
    #     if processor == None:
    #         processor = CLIPSimilarity.processor
    #     inputs = processor(text=text, images=image,
    #                        return_tensors="pt", padding=True)
    #     outputs = model(**inputs)
    #     # this is the image-text similarity score
    #     logits_per_image = outputs.logits_per_image
    #     # we can take the softmax to get the label probabilities
    #     probs = logits_per_image.softmax(dim=1).detach().numpy()[0]
    #     out = [(probs[i], text[i]) for i in (range(len(text)))]
    #     out.sort(reverse=True)
    #     filteredOut = np.array(out[:cutoff]).T
    #     return  pd.concat([pd.Series(filteredOut[0]),pd.Series(filteredOut[1])])#TODO FUCKING DISGUSTING CODE FIX IT
    @staticmethod
    def similarityScore(array: pd.Series, cutoff: int, model=None, processor=None):
        # TODO add memoization/pooling, use a more efficient datatype
        image = array[0]
        text = list(array[1:])
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
        out = [(probs[i], text[i]) for i in (range(len(text)))]
        out.sort(reverse=True)

        filteredOut = np.array(out[:cutoff]).T

        # TODO FUCKING DISGUSTING CODE FIX IT
        return pd.concat([pd.Series(filteredOut[0]), pd.Series(filteredOut[1])])


if __name__ == '__main__':  # testing code please ignore #TODO fix this to be up to date
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    image = Image.open(requests.get(url, stream=True).raw)
    text = [image, "a photo of two cats", "cats on a couch", "แมวนอนอยู่บนโซฟา",
            "two cats", "cats and remotes on a couch", "dogs on a couch", "ปลาทูแม่กลอง"]

    df = pd.DataFrame([text])
    newdf = df.apply(CLIPSimilarity.similarityScore, args=(5,), axis=1)
    
    print(newdf.head())
