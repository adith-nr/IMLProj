# HAR project workspace

Start with the [HARTH project guide](harth_project/README.md). It includes the four-member work distribution, study topics, notebook order, future run instructions and compute requirements.

- **Milestone:** install the dependencies once, then run notebooks 01 → 02 → 03 → 04. They directly perform data setup, feature preparation, classical training and error analysis through shared Python functions. No separate script commands are required.
- **Later:** notebooks 05–06 cover improvements and nested tuning, with training disabled by default.
- **Status:** pipeline prepared; experiments and pipeline tests have not been run. HARTH is already present in `harth_project/data/harth/`. No downloading is needed. The environment may still need installation completed.

This project uses **HARTH only**, under `harth_project/`. Notebook 01 reads the existing dataset; notebook 02 includes plots explaining the actual window splits and their features.

The notebooks depend on saved files from earlier notebooks, so run them in order. Separate kernels are fine; separate machines need the corresponding data/output files or must repeat the earlier notebooks. Re-running notebook 03's training cells retrains those models.

Functions used in only one notebook are defined there with explanations. Reused feature extraction, models and evaluation stay in `har_utils.py`. The optional Python script supports baseline training after notebooks 01–02.
