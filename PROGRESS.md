# Progress Report: Running SaProt, ProTrek, Pinal (and notes on Evolla)

**Author:** Md Maidul Islam
**Context:** Following up on Prof. Fajie Yuan's suggestion to study and run the group's open-source projects.

## A note on tools

I don't have a biology background, so for this exercise I used Claude Code (an AI coding assistant) as a
pair-programming tool — mainly to speed up environment setup, dependency debugging, and boilerplate code,
the same way I'd use an IDE with AI-assisted autocomplete. Every result below was something I ran, inspected,
and cross-checked myself (e.g. verifying the p53 mutation result against known cancer biology, verifying
token formats by hand). I'm flagging this upfront rather than leaving it implicit.

## Setup

No dedicated GPU. Used a MacBook Pro (M1, 17GB RAM). Google Colab's free T4 GPU was tried first for
SaProt but turned out unreliable (frequent disconnects), so I moved everything to local, CPU/MPS-based
inference instead.

---

## 1. SaProt

**What it does:** A protein language model that encodes both the amino acid sequence and a discretized
form of the 3D structure (Foldseek's "3Di" alphabet) in one combined token vocabulary, so the model can
reason about sequence and structure jointly.

**What I ran:**
- Basic masked-token inference with `SaProt_35M_AF2` (HuggingFace).
- A full structure-to-mutation-effect pipeline built from scratch: downloaded the Foldseek binary,
  extracted 3Di tokens from a real structure using `get_struc_seq()`, then implemented the paper's
  "masked marginal" zero-shot scoring method to score point mutations.
- **Validation:** tested the well-known cancer-associated p53 mutation R175H (AlphaFold structure for
  human TP53) — the model scored it **-1.39 (unfavorable)**, correctly flagging it as damaging, without
  any fine-tuning.

**Where I got stuck:** ColabSaprot's free GPU runtime kept disconnecting; I switched to running the
underlying model directly instead of relying on the notebook UI.

**Question for Prof. Yuan:** How sensitive is prediction quality to the accuracy of the input structure —
e.g. if the only available structure is a low-confidence AlphaFold prediction rather than an experimental
one, how much does that degrade downstream tasks like mutation scoring?

---

## 2. ProTrek

**What it does:** A trimodal model aligning protein sequence, structure, and natural-language function
descriptions in one embedding space via contrastive learning — enabling search across any pair of those
modalities.

**What I ran:**
- Sequence→text search: given human insulin's sequence and five candidate function descriptions, the
  model correctly ranked "a hormone that regulates blood glucose levels" highest by a wide margin.
- Structure→text search: repeated the same test using only p53's 3D structure (no amino acid sequence),
  and the model correctly identified it as "a tumor suppressor protein that prevents cancer by regulating
  the cell cycle" — confirming the structure-text alignment works independently of sequence.

**Where I got stuck:** the repo pins older `torchmetrics`/`transformers` versions with APIs that have
since changed; fixed with three small compatibility patches rather than downgrading the whole environment.

**Question for Prof. Yuan:** For protein families with very sparse function annotation, can the
sequence/structure modalities compensate during contrastive training, or does retrieval quality degrade
notably for poorly-annotated families?

---

## 3. Pinal

**What it does:** Text-guided de novo protein design — converts a natural-language description into a
protein structure (via a T2struc model), then that structure into an amino acid sequence (via a
fine-tuned SaProt variant, SaProt-T).

**What I ran:** the smallest available checkpoint combination (T2struc-1.2B + SaProt-T, ~7.5GB) locally.
Given the description "Actin.", the model generated a sequence whose N-terminal region closely matches
real human beta-actin — suggesting it drew on genuine structural/sequence knowledge rather than producing
arbitrary output.

**Where I got stuck (the most involved debugging of the four projects):**
- The repo's custom GPT-2 implementation depends on a `transformers` class (`SequenceSummary`) that's been
  removed in current versions — not fixable with a small patch, so I built a separate Python 3.10
  environment pinned to the repo's exact dependency versions.
- PyTorch's MPS backend (Apple GPU) doesn't support bfloat16, so inference ran on CPU (~9 min/protein).
- The SaProt-T checkpoint's pickle format required `deepspeed` to be importable (unused at inference time,
  but needed to unpickle) — resolved by installing the exact pinned version, since the latest one
  conflicts with the pinned PyTorch version.

**Question for Prof. Yuan:** Given that the "Actin" output closely resembled the real protein, how is
novelty vs. memorization evaluated for genuinely novel design targets (e.g. a hypothetical enzyme with no
close natural analog)?

---

## 4. Evolla

**What it does:** A protein-question-answering generative model combining a protein encoder with a large
language model decoder, so you can ask natural-language questions about a protein and get a text answer.

**Status: not yet run.** The smallest available checkpoint is 10B parameters (up to 80B for the larger
variant) — well beyond what a 17GB-RAM laptop with no dedicated GPU can load. I tried the hosted demo
(chat-protein.com) as an alternative, but it's currently returning a DNS/Cloudflare error on the server
side. I've read the paper and repo documentation and understand the architecture; I plan to retry the
hosted demo periodically.

**Question for Prof. Yuan:** Is there (or are there plans for) a smaller/distilled/quantized version of
Evolla usable on limited hardware, or is a large LLM decoder essentially required for this task?

---

## Overall reflection

Looking at all four together, they form more of a connected pipeline than four separate projects: the
"3Di" structure-tokenization idea from SaProt is reused directly inside ProTrek and inside Pinal's
SaProt-T component. There's also a natural progression — SaProt for understanding proteins, ProTrek for
searching them, Pinal for generating them, and Evolla for talking about them in natural language — each
building on the representations established by the earlier ones.

The biggest practical lesson was that published research code is rarely "pip install and run" a year or
two later — most of the real effort went into diagnosing and fixing environment/dependency mismatches
rather than the modeling logic itself, which is a skill I expect matters a lot for this kind of work.
