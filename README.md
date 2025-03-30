
# Empirical Copula Generator

## Overview
This repository introduces the Empirical Copula Generator, a novel non-parametric method designed to create synthetic datasets that preserve the statistical properties of original data while safeguarding privacy. This tool addresses a critical need for high-fidelity synthetic data in domains like healthcare, finance, and social sciences, where data scarcity and confidentiality often limit analysis. Through extensive experiments on both synthetic and real-world datasets—including Adult, Ecoli, Forest Fires, and Wisconsin Breast Cancer—the generator proves its ability to maintain marginal and joint distributions with exceptional accuracy, achieving a Jensen-Shannon Divergence as low as 0.0292 for the Wisconsin dataset. By delivering a robust and versatile solution for data augmentation and privacy-preserving research, this work bridges the gap between data availability and statistical integrity, enabling advanced machine learning applications and opening doors for future methodological advancements.

## Datasets and Code Files
The repository includes datasets and scripts essential to the Empirical Copula Generator. Below is a detailed breakdown:

### Real-World Datasets
- **`Forest Fires Dataset`**: Contains meteorological data for predicting forest fire occurrences.
- **`wdbc Dataset`**: The Wisconsin Diagnostic Breast Cancer dataset, widely used in medical research.
- **`ecoli`**: A dataset for classifying E. coli protein localization sites.
- **`census`**: The Adult dataset from the UCI repository, utilized for income prediction studies.

### Synthetic Dataset Generation
- **`createstar.py` and `augmentstar.py`**: Scripts to generate and augment star-shaped synthetic datasets.
- **`multiforms.py` and `augmentmultiforms.py`**: Scripts to create and augment multi-form synthetic datasets.

### Core Implementation
- **`Finalcode.py`**: The main implementation of the Empirical Copula Generator.
- **`optimizedcode.py`**: An optimized version (O2) of the generator for improved performance.

### Quality Assessment
- **`Assessmnetoptimalversion.py`**: A script to evaluate the quality of the synthetic data produced by the generator.

## Key Features
- **Privacy Preservation**: Ensures confidentiality of sensitive data.
- **Statistical Fidelity**: Maintains both marginal and joint distributions accurately.
- **Versatility**: Applicable to diverse domains and dataset types.

## Future Directions
- Benchmarking against methods like SMOTE, GANs, SDV-G, and VAEs.
- Improving scalability with parallelized implementations.
- Expanding support for time series and graph-based data.

## License
[Specify your license here, e.g., MIT, Apache 2.0, or CC BY 4.0]
