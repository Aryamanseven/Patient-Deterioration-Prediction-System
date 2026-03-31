# System Limitations & Future Works

PhysioGuard is a competition-tuned, comprehensive architecture. Still, several research and practical limitations stand:

## Current Limitations
1. **Memory Ceiling**: Real-world operations executing the `257` column time-series expansion (`engineer_features()`) will require memory batching when handling millions of patients simultaneously.
2. **Hard-Coded Categories**: Deep Models initialize Categorical Embeddings using predefined counts (e.g. `10` beds vs `500` devices). Unknown categorical labels in the live test stream will trigger Out-of-Bounds errors in PyTorch matrices.
3. **Synthetic Generative Bounds**: The `.data_loader()` currently features a synthetic placeholder. Distribution assumptions in this proxy don't adhere directly to complex multivariate probability distribution seen in reality.
4. **Hardware Bottleneck**: Deep Learning TCN-Transformer currently has hardcoded attention heads to 8; it might overwhelm VRAM on sub-12GB GPU architectures.

## Future Engineering
- Integrate Delta Lake / PySpark for distributed large-scale feature engineering (`engineer_features()`).
- Implement online streaming components using Apache Kafka to consume Vital monitors directly instead of static CSV files.
- Refactor the synthetic generation to implement a Conditional GAN to match exact patient trajectories for extreme edge-case simulation testing.
