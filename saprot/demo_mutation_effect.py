"""
Real mutation-effect prediction with SaProt, using an actual PDB/CIF structure.

Method: "masked marginal" zero-shot scoring (the approach SaProt/ESM papers use):
  1. Convert the structure to a combined AA+3Di token sequence via foldseek.
  2. Mask the position we want to mutate.
  3. Compare the model's predicted log-probability of the mutant token vs the
     wild-type token at that position. Positive score = mutation looks
     evolutionarily favorable to the model; negative = unfavorable.
"""
import utils.transport
import utils.session
import sys
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
from utils.foldseek_util import get_struc_seq

FOLDSEEK_BIN = "bin/foldseek"
CHAIN = "A"
MODEL_PATH = "westlake-repl/SaProt_35M_AF2"

# usage: python demo_mutation_effect.py <MUTATION> [structure_file]
# mutation format: "<wild-type AA><1-indexed position><mutant AA>", e.g. "N1A"
MUTATION = sys.argv[1] if len(sys.argv) > 1 else "N1A"
STRUCTURE_FILE = sys.argv[2] if len(sys.argv) > 2 else "example/8ac8.cif"


def parse_mutation(mut):
    wt_aa, pos, mt_aa = mut[0], int(mut[1:-1]), mut[-1]
    return wt_aa, pos, mt_aa


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Extracting AA + 3Di structure sequence from {STRUCTURE_FILE} (chain {CHAIN}) ...")
    aa_seq, struc_seq, combined_seq = get_struc_seq(FOLDSEEK_BIN, STRUCTURE_FILE, [CHAIN], plddt_mask=False)[CHAIN]
    print(f"Protein length: {len(aa_seq)}")
    print(f"AA sequence (first 40): {aa_seq[:40]}")
    print(f"3Di sequence (first 40): {struc_seq[:40]}")

    wt_aa, pos, mt_aa = parse_mutation(MUTATION)
    idx = pos - 1  # 0-indexed
    assert aa_seq[idx] == wt_aa, (
        f"Mismatch: position {pos} in structure is '{aa_seq[idx]}', not wild-type '{wt_aa}' you specified."
    )
    struc_char = struc_seq[idx].lower()
    wt_token = wt_aa + struc_char
    mt_token = mt_aa + struc_char
    print(f"\nMutation requested: {MUTATION}")
    print(f"Wild-type combined token at position {pos}: '{wt_token}'")
    print(f"Mutant combined token at position {pos}:    '{mt_token}'")

    print(f"\nLoading tokenizer and model from {MODEL_PATH} ...")
    tokenizer = EsmTokenizer.from_pretrained(MODEL_PATH)
    model = EsmForMaskedLM.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    wt_id = tokenizer.convert_tokens_to_ids(wt_token)
    mt_id = tokenizer.convert_tokens_to_ids(mt_token)
    if tokenizer.unk_token_id in (wt_id, mt_id):
        raise ValueError(
            f"Token not found in vocabulary (wt='{wt_token}' id={wt_id}, mt='{mt_token}' id={mt_id}). "
            "This can happen for rare AA+3Di combinations."
        )

    inputs = tokenizer(combined_seq, return_tensors="pt").to(device)
    # +1 because index 0 in input_ids is the <cls> token
    mask_pos = idx + 1
    original_token_id = inputs["input_ids"][0, mask_pos].item()
    inputs["input_ids"][0, mask_pos] = tokenizer.mask_token_id

    with torch.no_grad():
        logits = model(**inputs).logits[0, mask_pos]
        log_probs = torch.log_softmax(logits, dim=-1)

    wt_logp = log_probs[wt_id].item()
    mt_logp = log_probs[mt_id].item()
    score = mt_logp - wt_logp

    print(f"\nlog P(wild-type token)  = {wt_logp:.4f}")
    print(f"log P(mutant token)     = {mt_logp:.4f}")
    print(f"Mutation score (mt - wt) = {score:+.4f}")
    if score > 0:
        print("=> Model predicts this mutation is FAVORABLE (better than wild-type).")
    else:
        print("=> Model predicts this mutation is UNFAVORABLE (worse than wild-type).")


if __name__ == "__main__":
    main()
