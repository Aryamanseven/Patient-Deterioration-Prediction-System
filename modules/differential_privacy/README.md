# Differential Privacy Module

Provides optional DP-SGD integration through Opacus.

## Notes

- Works when backend supports required operations.
- On DirectML, DP-SGD is skipped (warning is logged).
- For strict DP guarantees, run on CUDA or CPU.

## Config section

- modules.differential_privacy
