# Westlake Protein AI — Exploration Notes

Small scripts and notes from running and validating four of Prof. Fajie Yuan's group's open-source
projects (SaProt, ProTrek, Pinal, Evolla), done locally on a MacBook Pro (M1) with no dedicated GPU.

See [PROGRESS.md](./PROGRESS.md) for the full write-up: what was run, what broke, how it was fixed, and
the results.

## Structure

```
saprot/   - SaProt inference + a from-scratch mutation-effect scoring pipeline
protrek/  - ProTrek sequence<->text and structure<->text search demos
pinal/    - Text-guided protein design with Pinal (T2struc + SaProt-T)
```

Each script assumes the corresponding upstream repo (cloned separately, not included here) as its
working directory, since they import from that repo's `model/` and `utils/` packages:

- https://github.com/westlake-repl/SaProt
- https://github.com/westlake-repl/ProTrek
- https://github.com/westlake-repl/Denovo-Pinal

## Notable results

- **SaProt:** correctly scored the cancer-associated p53 mutation R175H as unfavorable, using a
  from-scratch implementation of the paper's zero-shot masked-marginal scoring method.
- **ProTrek:** correctly matched human insulin to "a hormone that regulates blood glucose levels", and
  separately matched p53's 3D structure alone (no sequence) to "a tumor suppressor protein ...".
- **Pinal:** given the description "Actin.", generated a sequence whose N-terminus closely matches real
  human beta-actin.
