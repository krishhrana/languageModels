import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import tiktoken

from model import GPT2


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[GPT2, int | None]:
    model = GPT2().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key
        if cleaned_key.startswith('module.'):
            cleaned_key = cleaned_key.split('.', 1)[1]
        if cleaned_key.startswith('_orig_mod.'):
            cleaned_key = cleaned_key.split('.', 1)[1]
        cleaned_state_dict[cleaned_key] = value
    model.load_state_dict(cleaned_state_dict)
    return model.eval(), checkpoint.get('step')


@torch.no_grad()
def sample(model: GPT2, tokenizer, prompt: str, num_samples: int, max_tokens: int, temperature: float, top_k: int, device: torch.device):
    tokens = tokenizer.encode_ordinary(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0).repeat(num_samples, 1)
    torch.manual_seed(42)

    log_prob_sums = torch.zeros(num_samples, dtype=torch.float, device=device)
    generated_counts = torch.zeros(num_samples, dtype=torch.long, device=device)

    start_time = time.perf_counter()
    for _ in range(max_tokens):
        logits = model(tokens)
        curr_logits = logits[:, -1, :]
        scaled_logits = curr_logits / max(temperature, 1e-8)
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        probs = log_probs.exp()

        if top_k > 0:
            probs, top_idx = torch.topk(probs, k=top_k, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1)
            curr_tokens = torch.gather(top_idx, dim=-1, index=sampled_idx)
        else:
            curr_tokens = torch.multinomial(probs, num_samples=1)

        curr_log_probs = torch.gather(log_probs, dim=-1, index=curr_tokens)
        log_prob_sums += curr_log_probs.squeeze(-1)
        generated_counts += 1

        tokens = torch.cat([tokens, curr_tokens], dim=-1)

    elapsed = time.perf_counter() - start_time
    valid_counts = generated_counts.clamp_min(1).to(dtype=torch.float)
    avg_neg_log_likelihood = -log_prob_sums / valid_counts
    perplexities = avg_neg_log_likelihood.exp().cpu().tolist()

    decoded = tokenizer.decode_batch(tokens.cpu().tolist())
    return decoded, perplexities, elapsed


def parse_args():
    parser = argparse.ArgumentParser(description='Generate text from GPT-2 checkpoints.')
    parser.add_argument('--checkpoint', type=Path, default=Path('model_checkpoints/model_19072.pt'), help='Path to a checkpoint file.')
    parser.add_argument('--prompt', type=str, default='Hello I am GPT-2, a language model', help='Prompt to condition generation.')
    parser.add_argument('--num-samples', type=int, default=4, help='Number of samples to generate.')
    parser.add_argument('--max-tokens', type=int, default=64, help='Number of new tokens to sample.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature.')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling cutoff; set <=0 for full distribution.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device for generation.')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    tokenizer = tiktoken.encoding_for_model('gpt2')
    model, step = load_model(args.checkpoint, device)
    reported_step = step if step is not None else 'unknown'
    print(f'Loaded weights from {args.checkpoint} (step={reported_step})')

    samples, perplexities, elapsed = sample(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(f'{args.max_tokens} tokens per {args.num_samples} sequences took {elapsed:.2f} secs on {device}')
    for idx, ppl in enumerate(perplexities, start=1):
        print(f'Sample {idx} perplexity: {ppl:.4f}')

    for idx, text in enumerate(samples, start=1):
        print(f'--- Sample {idx} ---')
        print(text)
        print()


if __name__ == '__main__':
    main()
