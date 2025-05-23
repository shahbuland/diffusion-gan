from transformers import CLIPTextModel, CLIPTokenizer
import torch
from torch import nn

class TextEmbedder(nn.Module):
    def __init__(self, dim = None):
        super().__init__()

        self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def tokenize(self, text_list):
        return self.tokenizer(text_list, padding='max_length', max_length = 77, truncation=True, return_tensors="pt")

    def forward(self, input_ids, attention_mask):
        device = next(self.model.parameters()).device

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states = True)
        hidden_states = outputs.hidden_states[-2]  # Second last hidden state
        return hidden_states

    @torch.no_grad()
    def encode_text(self, text_list):
        return self.forward(**self.tokenize(text_list))