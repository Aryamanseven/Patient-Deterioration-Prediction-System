# Judge Q&A Defense Cheatsheet

Use this sheet after the pitch. Keep answers short, factual, and controlled.

## Q1: Why should we trust your results?

Answer:
"Because every key claim is tied to saved evidence: run metrics, predictions, benchmark summary, and reproducibility checks. We can point to files, not just slides."

## Q2: Why PR-AUC, not accuracy?

Answer:
"The class is imbalanced, around 5.4% positive. Accuracy can look high even with poor detection. PR-AUC is the correct metric for rare-event clinical risk detection."

## Q3: What is your strongest final metric?

Answer:
"The final focused model shows PR-AUC around 0.7396 with strong ROC-AUC support, and we keep run-level artifacts for independent verification."

## Q4: How did you avoid leakage?

Answer:
"We used episode-aware reconstruction and leakage-safe feature engineering. Validation split is group-aware by reconstructed episodes, not random row shuffling."

## Q5: Is this deployable right now?

Answer:
"It is deployment-oriented engineering, but not hospital-approved deployment. We position this as decision support requiring clinical governance."

## Q6: Why should your team win?

Answer:
"Because we combine three things in one package: measurable performance, reproducibility discipline, and a clean demo path. Reliable execution is our competitive edge."

## Q7: Did you use external data or APIs?

Answer:
"No external clinical data and no external prediction APIs were used in the core training workflow."

## Q8: What if someone cannot run your notebook?

Answer:
"We provide a reproducibility notebook with explicit Python requirement and evidence files. We also provide benchmark summaries and run artifacts for direct inspection."

## Q9: What are your system limitations?

Answer:
"Class imbalance remains challenging, episode IDs are inferred from provided fields, and this is not a substitute for clinical judgment."

## Q10: What is your final one-line position?

Answer:
"ANC-052 is a high-reliability deterioration prediction pipeline with evidence-backed performance and submission-ready reproducibility."

## Delivery rules during Q&A

1. Never guess numeric values from memory.
2. Keep each answer under 20 seconds.
3. If challenged, reference evidence files directly.
4. Never use absolute claims like always or guaranteed.
