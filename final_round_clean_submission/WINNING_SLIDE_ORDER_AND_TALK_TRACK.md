# Winning Slide Order and Talk Track (Aligned to Your 9-Slide Deck)

This script is synchronized to the current deck with 9 slides.

## Presentation objective

1. Establish credibility in first 30 seconds.
2. Show technical depth without overclaiming.
3. Anchor every performance claim to reproducible artifacts.
4. End with confidence, not hype.

## Slide-by-slide order and words

### Slide 1: Title and 12-hour warning claim (0:00 to 0:20)

Say:
"We are Team ANC-052. Our goal is a reliable 12-hour early warning system for patient deterioration from hourly clinical time-series. In the final round, we focused on what judges can trust most: measurable performance and strict reproducibility."

Do not say:
1. "Production deployed"
2. "Clinically approved"

### Slide 2: Problem context and motivation (0:20 to 0:45)

Say:
"The dataset is strongly imbalanced, only about 5.4% positive records, so plain accuracy is misleading. Our focus is early detection with controlled false alarms, where PR-AUC is the right metric for decision quality."

### Slide 3: Methodology and rationale (0:45 to 1:20)

Say:
"We reconstruct episodes, engineer leakage-safe temporal and clinical features, then train and compare model families. Final selection is evidence-based, using focused tuning and holdout validation rather than one-shot leaderboard luck."

### Slide 4: Dataset overview (1:20 to 1:40)

Say:
"We used only official provided files. Internal validation is episode-aware split to reflect realistic generalization. No external data, no hidden augmentation, no API shortcuts."

### Slide 5: Architecture workflow (1:40 to 2:00)

Say:
"The pipeline is input signals to feature extractor to classifier to risk output bands. This architecture is built for auditability: each stage leaves traceable artifacts for verification."

### Slide 6: Results (2:00 to 2:30)

Say:
"Our final tuned model reaches PR-AUC around 0.7396 with strong ROC-AUC support. More importantly, we maintain an evidence path from metrics to predictions to run logs, so claims are reproducible and reviewable."

### Slide 7: Challenges and resolution (2:30 to 2:55)

Say:
"We handled class imbalance, absent native episode IDs, and compute constraints by prioritized engineering choices: episode reconstruction, metric-correct optimization, and compact but complete artifact retention."

### Slide 8: Team intro (2:55 to 3:10)

Say:
"Our team split responsibilities across modeling, data pipeline, and interface integration so that research and execution quality moved together."

### Slide 9: Additional information and close (3:10 to 3:30)

Say:
"Final takeaway: ANC-052 is not a slide-only solution. It is a reproducible pipeline with evidence-backed results and a demo path aligned to clinical decision support boundaries."

## Voice control and body language

1. Keep pace at 145 to 160 words per minute.
2. Pause 1 second before every metric value.
3. Keep eye contact on transitions between slides 3, 6, and 9.
4. End with a full stop, not an upspeak question tone.

## Safe claim boundary

Allowed claims:
1. Reproducible internal evidence and validated metrics.
2. Decision-support utility for triage prioritization.

Not allowed claims:
1. Clinical deployment approval.
2. Guaranteed mortality reduction.
3. Outperforming all external models universally.
