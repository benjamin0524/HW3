import torch
import pickle
import numpy as np
import os
import argparse
from transformers import GPT2Config, GPT2LMHeadModel
import utils
import torch.nn as nn

X_LEN = 512 

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_path', type=str, help='The dictionary path', default='./dictionary.pkl')
    parser.add_argument('--device', type=str, help='Device to run the model on', default='cuda')
    parser.add_argument('--model_path', type=str, help='Path to the trained model checkpoint', required=True)
    parser.add_argument('--prompt_path', type=str, help='Path to the prompt MIDI file (8 bars)', required=True)
    parser.add_argument('--output_path', type=str, help='Path to save the generated continuation MIDI', required=True)
    parser.add_argument('--n_target_bar', type=int, help='Number of bars to generate', default=24)
    parser.add_argument('--temperature', type=float, help='Sampling temperature', default=1.2)
    parser.add_argument('--topk', type=int, help='Top-k for sampling', default=5)
    args = parser.parse_args()
    return args

class Model(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_head=12, n_layer=12, d_inner=2048):
        super(Model, self).__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=X_LEN, 
            n_ctx=X_LEN, 
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
            use_cache=False,
        )
        self.transformer = GPT2LMHeadModel(config)

    def forward(self, x):
        outputs = self.transformer(input_ids=x)
        logits = outputs.logits  
        return logits

def temperature_sampling(logits, temperature, topk):
    """
    Implement temperature sampling with top-k
    """
    logits = logits / temperature
    
    if topk > 0:
        topk_logits, topk_indices = torch.topk(logits, min(topk, len(logits)), dim=-1)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, topk_indices, topk_logits)
        logits = mask

    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    if probs.sum() == 0:
        probs = torch.ones_like(probs) / len(probs)
    else:
        probs = probs / probs.sum()
    
    word = torch.multinomial(probs, 1).item()
    return word

def test_continuation(prompt_path, n_target_bar=24, temperature=1.2, topk=5, output_path='', model_path=''):
    """
    Continuation generation for Task 2
    """

    event2word, word2event = pickle.load(open(args.dict_path, 'rb'))
    device = args.device


    note_items, tempo_items = utils.read_items(prompt_path)
    note_items = utils.quantize_items(note_items)
    max_time = note_items[-1].end
    
    items = tempo_items + note_items
    groups = utils.group_items(items, max_time)
    events = utils.item2event(groups)


    words = []
    for event in events:
        e = '{}_{}'.format(event.name, event.value)
        if e in event2word:
            words.append(event2word[e])
        else:
            if event.name == 'Note Velocity':
                if 'Note Velocity_21' in event2word:
                    words.append(event2word['Note Velocity_21'])
                else:
                    velocity_keys = [k for k in event2word.keys() if 'Note Velocity' in k]
                    if velocity_keys:
                        words.append(event2word[velocity_keys[0]])

    prompt_bars = words.count(event2word['Bar_None'])
    print(f"Prompt contains {prompt_bars} bars")


    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    vocab_size = checkpoint.get('vocab_size', len(event2word))
    model = Model(vocab_size=vocab_size).to(device)
    

    state_dict = checkpoint['model']
    state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

 
    current_generated_bar = 0
    max_tokens = n_target_bar * 500  
    token_count = 0
    
    print(f'Start generating continuation for {n_target_bar} bars...')
    while current_generated_bar < n_target_bar and token_count < max_tokens:
        token_count += 1
        if token_count % 100 == 0:
            print(f"\rGenerated {current_generated_bar}/{n_target_bar} bars, tokens: {token_count}", end="")
        

        if len(words) > X_LEN:
            temp_x = torch.tensor([words[-X_LEN:]], dtype=torch.long).to(device)
        else:
            temp_x = torch.tensor([words], dtype=torch.long).to(device)
        

        output_logits = model(temp_x)
        

        _logit = output_logits[0, -1].cpu().detach()
        word = temperature_sampling(_logit, temperature, topk)
        
        words.append(word)
        
        if word == event2word['Bar_None']:
            current_generated_bar += 1
    
    print(f"\nGeneration completed with {len(words)} tokens")
    utils.write_midi(
        words=words,
        word2event=word2event,
        output_path=output_path,
        prompt_path=prompt_path
    )
    print(f"Saved generated MIDI to: {output_path}")

if __name__ == '__main__':
    args = parse_opt()
    test_continuation(
        prompt_path=args.prompt_path,
        n_target_bar=args.n_target_bar,
        temperature=args.temperature,
        topk=args.topk,
        output_path=args.output_path,
        model_path=args.model_path
    )
