"""
Saliency / explanation tooling for the disagreement study (paper framing 1).

Modules:
  - saliency_metrics: pairwise disagreement metrics across explanations
    (feature agreement, rank agreement, sign agreement, rank correlation,
    top-k Jaccard). Implements the family from Krishna et al. 2022 and
    Jukic et al. ACL 2023 Findings.
  - phenomena: bucket sentences by linguistic phenomena (OOV-rate,
    multi-word entities, abbreviation patterns) so the disagreement
    can be conditioned on what the biomedical domain actually contains.
  - faithfulness: deletion / insertion AUC, computed against any model
    that exposes a P(target | tokens) callable. Used as a sanity check
    that domain-shifted disagreement is not just one method failing.
  - integrated_gradients: IG for HuggingFace token-classification
    models (Captum). The third saliency method alongside LIME and
    attention.
"""
