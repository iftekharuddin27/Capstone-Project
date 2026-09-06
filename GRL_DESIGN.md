# Future GRL Robustness Experiment — Design Only

Status: optional future experiment; not implemented, reproduced, or authorized to run

## Purpose and ordering

A gradient reversal layer (GRL) could test whether representations can be made less predictive of Bangla corpus source. It must not be presented as part of the historical model. It should be attempted only after corrected single-task and gold-axis baselines run successfully, because the source variable is strongly confounded with the class label.

## Proposed five-source mapping

This is a new deterministic design mapping, not a recovered historical mapping:

| Domain ID | Bangla source |
|---:|---|
| 0 | ALERT |
| 1 | BD_SHS |
| 2 | BenSarc |
| 3 | BanglaSarc3 |
| 4 | BIDWESH |

Persist this mapping in every future run configuration and checkpoint metadata. Unknown sources must cause validation failure rather than being mapped to a catch-all class.

## Proposed architecture

```text
text -> shared encoder -> pooled representation -> task head(s)
                                      |
                                      +-> GRL -> domain MLP -> 5-source logits
```

The GRL is identity in the forward pass. During backpropagation it multiplies the encoder-side domain gradient by `-lambda_grl`. The domain head itself receives the ordinary gradient required to predict source.

A minimal domain head is `Linear(hidden, 256) -> ReLU -> Dropout -> Linear(256, 5)`. Domain loss is five-class cross entropy using the fixed source mapping. The total objective is:

```text
L_total = L_task + lambda_aux * L_aux + lambda_domain * L_domain
```

`lambda_domain` weights the domain objective; `lambda_grl` scales only its reversed encoder gradient. Record both separately.

## GRL coefficient schedule

For normalized training progress `p` in `[0,1]`, use the standard gradual schedule:

```text
lambda_grl(p) = 2 / (1 + exp(-10 * p)) - 1
```

It begins near zero and approaches one. Log the coefficient per optimizer step. A constant or alternative schedule would be a separate configured experiment.

## Required controls and reporting

- Start from the same corrected baseline configuration and data hashes.
- Keep model/tokenizer revisions, seed, batches, optimizer, scheduler, and selection metric fixed.
- Compare task-only baseline, domain head without reversal, and GRL model.
- Select checkpoints only by validation task macro F1; never by test performance.
- Report task metrics overall and by source, domain accuracy, per-source support, and worst-source performance.
- Run source-stratified diagnostics and examine whether rare ALERT/BIDWESH sources dominate variance.
- Add later multi-seed support for 13, 21, 42, 87, and 101 on Kaggle only.

## Confounding warning

The released Bangla sources are not label-balanced: ALERT is only class 0, BanglaSarc3 only class 2, BenSarc only classes 0/2, and hateful samples occur only in BD_SHS/BIDWESH. Suppressing source information can therefore suppress genuine label-predictive information, and low domain accuracy does not prove removal of dataset artifacts. Conversely, high task performance may still rely on residual source cues.

Because some source/label combinations do not exist, GRL cannot recover counterfactual evidence that was never collected. The preferred remedy is better cross-source annotation/data collection; GRL is only a robustness probe.

## Preconditions

Do not implement or run this design until corrected non-GRL baselines are verified and the supervisor approves the source mapping, domain-loss weight, evaluation design, and interpretation limits.

