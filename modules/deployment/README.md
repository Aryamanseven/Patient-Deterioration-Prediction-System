# Deployment Module

Exports trained artifacts for inference workflows.

## Outputs

- model/model.cbm (CatBoost)
- model/dl_model.onnx (optional ONNX export)

## Notes

- Controlled by modules.deployment.enabled.
- Enable for export-focused runs; disable for faster training-only runs.

## Config section

- modules.deployment
