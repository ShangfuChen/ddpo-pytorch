
import os
import argparse
import numpy as np

from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch

"""
If loading the original PickScore model:
    model = AutoModel.from_pretrained(pretrained_model_name_or_path="yuvalkirstain/PickScore_v1").eval().to(device)

If loading from checkpoint:
    model = AutoModel.from_pretrained(ckpt_path).eval().to(device)

    where ckpt_path = "checkpoint-final", given the following file structure:
        checkpoint-final
        |_ config.json
        |_ pytorch_model.bin
        |_ training_stage.json
"""

def calc_probs(processor, model, images, prompt, device):
    # processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    # processor = AutoProcessor.from_pretrained(processor_name_or_path)

    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)


    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        # get probabilities if you have multiple images to choose from
        probs = torch.softmax(scores, dim=-1)
    
    return probs.cpu().tolist(), scores.cpu().tolist()


def main(args):

    device = "cuda"

    ### Load model from checkpoint ###
    model = AutoModel.from_pretrained(args.ckpt_path).eval().to(device)

    scores = {}
    imgs = []
    img_names = os.listdir(args.img_dir)
    for img in img_names:
        # image inputs to the PickScore model is a list of PIL.Image
        imgs.append(Image.open(os.path.join(args.img_dir, img)))

    # get probabilities and scores for input images given a prompt
    probs, scores = calc_probs(model=model, images=imgs, prompt=args.prompt, device=device)

    print("Scores: ", scores)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--img-dir", type=str)
    parser.add_argument("--ckpt-path", type=str)
    args = parser.parse_args()
    main(args)
