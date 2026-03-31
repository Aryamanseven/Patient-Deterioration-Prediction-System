# Self-Supervised Learning (SSL) Module

This module implements self-supervised pre-training, giving our deep learning models a massive performance boost by learning the underlying structure of physiological data *before* seeing any labels.

## Method: Masked Sequence Prediction
Similar to BERT in NLP or masked autoencoders in vision.
1. We take sliding windows of patient vitals.
2. We randomly mask out `mask_ratio` (e.g., 15%) of the timesteps.
3. A Transformer encoder attempts to reconstruct the missing continuous values.
4. We compute MSE loss between predictions and the actual masked values.

## Artifacts Produced
- `ssl_encoder_weights.pt`: The initial weights that the main `TCNTransformer` or `LSTMAttention` model will load before supervised fine-tuning.

## Config
Controlled via `modules.ssl` in `default.yaml`.
