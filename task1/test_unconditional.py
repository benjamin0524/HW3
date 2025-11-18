import torch
import os
import numpy as np
from tqdm import tqdm
from torch import nn
import pickle
from transformers import GPT2Config, GPT2LMHeadModel
import utils

def sample_with_temperature(logits, temp, topk):
    logits = logits / temp
    if topk > 0:
        topk_values, topk_indices = torch.topk(logits, min(topk, len(logits)), dim=-1)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, topk_indices, topk_values)
        logits = mask
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.multinomial(probs / probs.sum(), 1).item() if probs.sum() > 0 else torch.randint(len(probs), (1,)).item()

def generate_music(n_bars=32, temp=1.2, topk=5, output_file='', model_file='', num_samples=1, dict_file='./dictionary.pkl'):
    event2word, word2event = pickle.load(open(dict_file, 'rb'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with torch.no_grad():
        checkpoint = torch.load(model_file, map_location=device, weights_only=False)
        vocab_size = checkpoint.get('vocab_size', len(event2word))
        model = Model(vocab_size=vocab_size).to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        for sample_idx in range(num_samples):
            words = [event2word['Bar_None']]
            tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
            tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
            if tempo_classes and tempo_values:
                words.extend([event2word['Position_1/16'], np.random.choice(tempo_classes), np.random.choice(tempo_values)])
            
            bars_generated = 0
            token_count = 0
            max_tokens = n_bars * 500
            print(f'Generating sample {sample_idx + 1}/{num_samples}...')
            
            while bars_generated < n_bars and token_count < max_tokens:
                token_count += 1
                if token_count % 100 == 0:
                    print(f"\rGenerated {bars_generated}/{n_bars} bars, tokens: {token_count}", end="")
                
                temp_x = torch.tensor([words[-512:]], dtype=torch.long).to(device) if len(words) > 512 else torch.tensor([words], dtype=torch.long).to(device)
                output_logits = model(temp_x)
                word = sample_with_temperature(output_logits[0, -1].cpu().detach(), temp, topk)
                words.append(word)
                if word == event2word['Bar_None']:
                    bars_generated += 1

            print(f"\nGeneration complete, {len(words)} tokens produced.")
            save_path = output_file if num_samples == 1 else os.path.join(os.path.dirname(output_file), f"{os.path.splitext(os.path.basename(output_file))[0]}_{sample_idx+1}.mid")
            utils.write_midi(words=words, word2event=word2event, output_path=save_path, prompt_path=None)
            print(f"Saved to: {save_path}")

class Model(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_head=12, n_layer=12, d_inner=2048):
        super(Model, self).__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=512,
            n_ctx=512,
            n_embd=d_model,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=d_inner,
            activation_function="gelu",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=False
        )
        self.transformer = GPT2LMHeadModel(config)
        
    def forward(self, x):
        return self.transformer(input_ids=x).logits

def main():
    output_path = './generated.mid'
    model_path = './best_model.pkl'
    dict_path = './dictionary.pkl'
    
    generate_music(
        n_bars=32,
        temp=1.2,
        topk=5,
        output_file=output_path,
        model_file=model_path,
        num_samples=20,
        dict_file=dict_path
    )

if __name__ == '__main__':
    main()
