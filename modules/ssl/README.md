# Self-Supervised Learning (SSL) Module

Implements masked sequence pretraining and SSL weight handoff.

## What it does

1. Masks a fraction of sequence timesteps.
2. Trains encoder to reconstruct masked values.
3. Saves reusable SSL weights for downstream DL training.

## Current run behavior

- In reuse-enabled profiles, SSL pretraining is skipped and existing weights are reused.
- SSL weights are copied into each run folder for provenance.

## Primary artifact

- ssl_pretrained_tcntransformer.pt

## Config section

- modules.ssl
