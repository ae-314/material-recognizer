# material-recognizer
Material recognition &amp; semantic texture retrieval with CLIP RN50 + Spark ML (OvR LinearSVC), tracked in MLflow.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iBX3A4-GTIHQWBkqZrYjXbwVeZT-v4rq)



# Abstract
We build a material recognition and semantic retrieval pipeline over large texture corpora using **CLIP RN50** embeddings and a **linear head** trained in **Apache Spark**. Images from **MINC-2500** (23 material classes) and **DTD** (47 describable attributes) are embedded into a 1024-D space. A **One-vs-Rest LinearSVC** head trained on MINC achieves **0.7057 accuracy** and **0.7008 weighted-F1** on a held-out split. We also demonstrate **text→image retrieval** by encoding natural-language prompts with CLIP’s text encoder and ranking images by cosine similarity, producing contact sheets for artist support and reference search.

# Data
- **MINC-2500**: ~57.5k material images across 23 classes. Used for **supervised** training/evaluation of the classifier head.  
- **DTD (Describable Textures Dataset)**: ~5.6k images across 47 attribute categories (e.g., *striped, dotted, wrinkled*). Used for **semantic retrieval** demonstrations.  
- Both datasets are stored as **Parquet** with columns `image.bytes`, `image.path`, `label`, plus derived columns: `embedding` (float32[1024]), `sha` (SHA-1 of bytes).  
- Embeddings for both sets are produced with **CLIP RN50** and **L2-normalized** per vector.

# Methods
**Embeddings.** Images are embedded with CLIP RN50 (PyTorch). Each embedding is L2-normalized to stabilize cosine similarity and linear probing.

**Classifier head.** We train a **One-vs-Rest LinearSVC** (hinge loss) in **Spark ML** over MINC embeddings:  
1) Load Parquet into Spark DataFrames.  
2) Convert `embedding: array<float>` to Spark’s `VectorUDT` features.  
3) Deterministic split (80/20).  
4) Fit LinearSVC (`maxIter=200`, `regParam=0.02`).  
5) Evaluate with `MulticlassClassificationEvaluator` (accuracy, weighted-F1).

**Semantic retrieval.** For a text prompt, CLIP’s **text encoder** produces a normalized vector. For each image embedding, relevance is the dot product (cosine similarity). Top-K images are returned and displayed as a contact sheet. Dataset labels are reported **post-hoc** for interpretability only.

**Experiment tracking.** All runs are logged to **MLflow (Machine Learning flow)** with parameters, metrics, artifacts (confusion matrix, contact sheets), and the saved Spark model under a Drive-backed tracking URI.

# Results
- **Linear head (MINC):** accuracy **0.7057**, weighted-F1 **0.7008** on 11,369 validation images; the confusion matrix shows a strong diagonal and symmetric confusions between visually similar materials.  
- **Text→image retrieval (DTD):** prompts such as *“striped, high-contrast fabric”* return images dominated by *striped/banded/lined*, demonstrating attribute-level alignment without using labels during search.

# Reproducibility and Artifacts
- Embedding Parquet files and the trained Spark model are stored in project Drive.  
- MLflow experiment `material_head_rn50` contains runs, metrics, confusion-matrix PNGs, and the `spark_model` artifact.  
- Retrieval grids for both MINC and DTD are saved to Drive and logged to MLflow.

# Limitations
- Linear heads saturate when classes overlap in the embedding space; larger gains require fine-tuning or a shallow nonlinearity.  
- Attribute names may be semantically close (e.g., *striped* vs *banded*), leading to mixed top-K distributions that still match visual intent.

# Usage and Demo Guidance
- **Prompts:** Any natural language is valid; CLIP generalizes beyond dataset label names. If results seem off, broaden or rephrase (e.g., *“coarse stone surface”* instead of *“granite cobblestone alley at dusk”*).  
- **Feedback:** Show top-K previews and simple label histograms so users can judge match quality.  
- **Optional widget:** Provide a prompt box and K selector, plus a toggle for **raw nearest neighbors** vs **class-constrained neighbors** (predict a material with the head, then retrieve within that class). Offer a suggestion list using DTD’s attribute names to set expectations.

