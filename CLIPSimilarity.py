from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from typing import List

class CLIPSimilarity:  # implementation class for CLIP similarithy

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    @staticmethod
    # takes a PIL image and a list of text and outputs the softmax similarity for each entry of text
    def similarityScore(image: Image, text: List[str], model=None, processor=None):
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

        return {text[i]: probs[i] for i in (range(len(text)))} #outputs the probability as a key/value dict
    

if __name__=='__main__':#testing code please ignore
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    text = ["a photo of two cats", "cats on a couch","a cat on a couch","two cats","cats and remotes on a couch"]
    print(CLIPSimilarity.SimilarityScore(image, text))
    image.show()
