from transformers import AutoTokenizer, AutoModel
import torch
import os
import json
import glob
import sys
import numpy as np

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")


def extract_geologic_text(data):
    map_data = data["success"]["data"]["mapData"][0]

    name = map_data.get("name", "")
    age = map_data.get("age", "")
    strat_name = map_data.get("strat_name", "")
    lith = map_data.get("lith", "")
    descrip = map_data.get("descrip", "")

    lithologic_info = []
    for lith_entry in map_data.get("liths", []):
        lith_name = lith_entry.get("lith", "")
        lith_type = lith_entry.get("lith_type", "")
        lith_class = lith_entry.get("lith_class", "")
        lithologic_info.append(f"{lith_name} ({lith_type}, {lith_class})")

    lithologic_info_str = "; ".join(lithologic_info)

    return f"{name}. Age: {age}. Stratigraphy: {strat_name}. Lithology: {lith}. Description: {descrip}. Lithologic components: {lithologic_info_str}."


def get_embedding(text: str):
    max_length = 512
    # Tokenize the entire text first to get token IDs
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"].squeeze(0)  # remove batch dimension

    # Split into chunks of max_length - 2 (to reserve space for [CLS] and [SEP] tokens)
    chunk_size = max_length - 2
    num_chunks = (input_ids.size(0) + chunk_size - 1) // chunk_size
    chunk_embeddings = []

    for i in range(num_chunks):
        # Extract a chunk of input_ids
        chunk_ids = input_ids[i * chunk_size : (i + 1) * chunk_size]

        # Add [CLS] and [SEP] tokens
        chunk_ids = torch.cat(
            [
                torch.tensor([tokenizer.cls_token_id]),
                chunk_ids,
                torch.tensor([tokenizer.sep_token_id]),
            ]
        )

        # Convert to tensor and get attention mask
        chunk_input = {
            "input_ids": chunk_ids.unsqueeze(0),
            "attention_mask": torch.ones_like(chunk_ids).unsqueeze(0),
        }

        # Pass the chunk through the model
        with torch.no_grad():
            outputs = model(**chunk_input)

        # Mean pooling to get the chunk embedding
        attention_mask = chunk_input["attention_mask"]
        masked_embeddings = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
        chunk_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(
            dim=1, keepdim=True
        )
        chunk_embeddings.append(chunk_embedding)

    # Concatenate all chunk embeddings and average them to get the final sentence embedding
    sentence_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0)
    return sentence_embedding.squeeze().numpy()


def write_embedding(embedding, file_name):
    new_name = file_name.replace("lithology/", "embeddings/lithology/").replace(
        ".json", ".npy"
    )
    new_directory = os.path.dirname(new_name)
    os.makedirs(new_directory, exist_ok=True)
    np.save(new_name, embedding)


if __name__ == "__main__":
    # directory = Config.DATA_DIR_LBL_LITH
    directory = Config.DATA_DIR_UNLBL_LITH

    for filename in glob.glob(os.path.join(directory, "*.json")):
        with open(filename, encoding="utf-8", mode="r") as currentFile:
            data = json.loads(currentFile.read())
            description = extract_geologic_text(data)
            embedding = get_embedding(description)
            write_embedding(embedding, filename)
