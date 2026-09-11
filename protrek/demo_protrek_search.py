"""
Small local demo of ProTrek: given one protein sequence and several candidate
function descriptions (natural language text), find which description matches
the protein best -- this is the core "sequence <-> function search" ability.
"""
import torch
import torchmetrics
import functools

# The repo pins an old torchmetrics API (Accuracy() with no args). Newer
# torchmetrics requires a `task` argument. We only need embeddings for this
# demo (no training/metrics computation), so patch it to be permissive.
torchmetrics.Accuracy = functools.partial(torchmetrics.Accuracy, task="multiclass", num_classes=2)

# Newer torch removed the `verbose` positional arg from _LRScheduler.__init__,
# but the repo's ConstantLRScheduler (used as the default, unused-in-inference
# scheduler) still passes it. Patch it to drop that arg.
import utils.lr_scheduler as lr_scheduler_module

def _patched_constant_init(self, optimizer, last_epoch=-1, verbose=False, init_lr=0.):
    self.init_lr = init_lr
    lr_scheduler_module._LRScheduler.__init__(self, optimizer, last_epoch)

lr_scheduler_module.ConstantLRScheduler.__init__ = _patched_constant_init

# Newer transformers renamed/removed PreTrainedTokenizerBase.batch_encode_plus
# in favor of just calling the tokenizer directly. Patch it back in.
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, "batch_encode_plus"):
    def _batch_encode_plus(self, texts, **kwargs):
        return self(texts, **kwargs)
    PreTrainedTokenizerBase.batch_encode_plus = _batch_encode_plus

from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel

WEIGHTS_DIR = "weights/ProTrek_35M_UniRef50"
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

config = {
    "protein_config": f"{WEIGHTS_DIR}/esm2_t12_35M_UR50D",
    "text_config": f"{WEIGHTS_DIR}/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "structure_config": f"{WEIGHTS_DIR}/foldseek_t12_35M",
    "from_checkpoint": f"{WEIGHTS_DIR}/ProTrek_35M.pt",
}

print("Loading ProTrek_35M model ...")
model = ProTrekTrimodalModel(**config).eval().to(device)

# Human insulin (a very well known protein) sequence (mature chains B+A concatenated for demo)
protein_seq = "FVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"

candidate_texts = [
    "A hormone that regulates blood glucose levels",
    "A protein involved in DNA repair",
    "A structural protein found in hair and nails",
    "An enzyme that digests starch into sugars",
    "A photosynthesis-related protein found in plants",
]

with torch.no_grad():
    seq_emb = model.get_protein_repr([protein_seq])
    text_emb = model.get_text_repr(candidate_texts)
    scores = (seq_emb @ text_emb.T / model.temperature).squeeze(0)

print(f"\nProtein sequence (insulin, {len(protein_seq)} aa):\n{protein_seq}\n")
print("Similarity scores against candidate function descriptions:")
ranked = sorted(zip(candidate_texts, scores.tolist()), key=lambda x: -x[1])
for text, score in ranked:
    print(f"  {score:+7.3f}  |  {text}")

print(f"\nBest match: \"{ranked[0][0]}\"")
