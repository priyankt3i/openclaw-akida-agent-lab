import argparse
import json
from pathlib import Path

import numpy as np

ARTIFACT_DIR = Path('prototype-1/artifacts')
DEFAULT_OUT = ARTIFACT_DIR / 'temporal_path_eval_contract_check.json'


def rel_mse(reference, candidate):
    diff = candidate - reference
    return float(np.mean(diff ** 2) / (np.mean(reference ** 2) + 1e-12))


def token_rel_mse(reference, candidate):
    diff = candidate - reference
    return np.mean(diff ** 2, axis=(0, 2)) / (np.mean(reference ** 2, axis=(0, 2)) + 1e-12)


def slice_rel_mse(reference, candidate, start, end):
    if end <= start:
        return None
    return rel_mse(reference[:, start:end, :], candidate[:, start:end, :])


def build_report(reference, candidate, fifo_depth, toolchain_ok):
    if reference.shape != candidate.shape or reference.ndim != 3:
        raise ValueError('reference and candidate must both have shape [batch, tokens, dim]')

    tokens = reference.shape[1]
    warmup_tokens = max(fifo_depth - 1, 0)
    steady_start = min(warmup_tokens, tokens)
    transition_end = min(warmup_tokens + fifo_depth, tokens)

    per_token = token_rel_mse(reference, candidate)
    warmup_rel = slice_rel_mse(reference, candidate, 0, steady_start)
    transition_rel = slice_rel_mse(reference, candidate, steady_start, transition_end)
    steady_rel = slice_rel_mse(reference, candidate, steady_start, tokens)

    steady_per_token = per_token[steady_start:]
    if steady_per_token.size >= 4:
        quarter = max(1, steady_per_token.size // 4)
        early = float(np.mean(steady_per_token[:quarter]))
        late = float(np.mean(steady_per_token[-quarter:]))
        late_to_early = float(late / max(early, 1e-12))
    elif steady_per_token.size > 0:
        early = float(np.mean(steady_per_token))
        late = float(np.mean(steady_per_token))
        late_to_early = 1.0
    else:
        early = None
        late = None
        late_to_early = None

    warmup_penalty_ratio = None
    if warmup_rel is not None and steady_rel is not None:
        warmup_penalty_ratio = float(warmup_rel / max(steady_rel, 1e-12))

    gates = {
        'toolchain_ok': bool(toolchain_ok),
        'enough_tokens_for_steady_state': tokens >= fifo_depth + 8,
        'steady_state_rel_mse_le_0p30': steady_rel is not None and steady_rel <= 0.30,
        'late_to_early_ratio_le_1p25': late_to_early is not None and late_to_early <= 1.25,
        'warmup_penalty_ratio_le_1p50': warmup_penalty_ratio is not None and warmup_penalty_ratio <= 1.50,
    }
    gates['pass'] = all(gates.values())

    return {
        'contract_version': '2026-04-08-temporal-path-v1',
        'fifo_depth': fifo_depth,
        'warmup': {
            'policy': 'Zero-prime the FIFO and treat exactly the first fifo_depth - 1 tokens as warmup. Report them, but exclude them from the continuation gate except through warmup_penalty_ratio.',
            'warmup_token_count': warmup_tokens,
            'steady_state_start_token': steady_start,
            'transition_end_token_exclusive': transition_end,
        },
        'metrics': {
            'full_sequence_rel_mse': rel_mse(reference, candidate),
            'warmup_rel_mse': warmup_rel,
            'transition_rel_mse': transition_rel,
            'steady_state_rel_mse': steady_rel,
            'steady_state_early_quarter_rel_mse': early,
            'steady_state_late_quarter_rel_mse': late,
            'steady_state_late_to_early_ratio': late_to_early,
            'warmup_penalty_ratio': warmup_penalty_ratio,
            'per_token_rel_mse': per_token.tolist(),
        },
        'gates': gates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', type=Path, required=True)
    parser.add_argument('--reference-key', default='reference')
    parser.add_argument('--candidate-key', required=True)
    parser.add_argument('--fifo-depth', type=int, required=True)
    parser.add_argument('--toolchain-ok', action='store_true')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    data = np.load(args.npz)
    reference = np.asarray(data[args.reference_key], dtype=np.float32)
    candidate = np.asarray(data[args.candidate_key], dtype=np.float32)
    report = build_report(reference, candidate, args.fifo_depth, args.toolchain_ok)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
