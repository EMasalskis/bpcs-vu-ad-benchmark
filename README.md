# DL Drug Repositioning Benchmark for Alzheimer's Disease

This repository contains the code for the bachelor's thesis "Literature Review and Evaluation of Deep Learning-Based Drug Repositioning Platforms". It provides a framework for running and evaluating two state-of-the-art drug repositioning models, **HNNDTA** and **DRML-Ensemble**, with a focus on Alzheimer's Disease.

## Project Structure

The repository is organized to separate data, model implementations, and the main benchmarking logic.

```
bpcs-benchmark/
│
├── data_input/ # Evaluation input files here
│ └── my_drug_list.csv
│
├── data_output/ # Benchmark results are saved here
│ ├── HNNDTA/
│ └── DRML-Ensemble/
│
├── DRML-Ensemble/ # Source code and data for the DRML-Ensemble
│ └── benchmark_drml_ensemble.py
│
├── HNNDTA/ # Source code and data for the HNNDTA
│ └── benchmark_hnndta.py
│
├── benchmark.py # The main script to run all benchmarks
├── requirements.txt # Python package requirements
└── README.md # This file
```

## Installation

This project uses `conda` for environment management.

1.  **Create and Activate Conda Environment**
    ```
    conda create -n dl_benchmark_env python=3.9
    conda activate dl_benchmark_env
    ```

2.  **Install PyTorch (CUDA 11.8)**
    ```
    conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

3.  **Install DGL (CUDA 11.8)**
    ```
    pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
    ```

4.  **Install Other Requirements**
    ```
    pip install -r requirements.txt
    ```

## Usage

The framework supports both training the original models and running a unified benchmark on a custom dataset.

### Training (Models are untracked)

To retrain the models from scratch using their original training data, navigate to the specific model's directory and run its training script.

-   **HNNDTA:**
    ```bash
    cd HNNDTA
    python model_train.py
    ```
-   **DRML-Ensemble:**
    ```bash
    cd DRML-Ensemble
    python main.py
    ```

### Running the Benchmark

The primary way to use this repository is through `benchmark.py` script located in the project root.

1.  **Prepare Your Input File**
    -   Create a `.csv` file inside the `data_input/` directory (e.g., `data_input/my_drug_list.csv`).
    -   The file must have at least two columns with no header:
        -   **Column 0:** DrugBank ID (e.g., `DB00915`)
        -   **Column 1:** Isomeric SMILES string (e.g., `C1CC(C(N1)C(=O)O)C(=O)O`)

2.  **Execute the Benchmark Script**
    From the project root directory (`bpcs-benchmark/`), run the following command:

    ```bash
    python benchmark.py [MODEL_NAME] [INPUT_FILE_PATH]
    ```

    **Examples:**

    -   To run the benchmark for **HNNDTA**:
        ```bash
        python benchmark.py HNNDTA data_input/my_drug_list.csv
        ```

    -   To run the benchmark for **DRML-Ensemble**:
        ```bash
        python benchmark.py DRML-Ensemble data_input/my_drug_list.csv
        ```

    The results will be saved to the corresponding folder in `data_output/`.

## Credits

This project adapts and builds upon the work of the original authors. The source code for the models was obtained from their respective repositories:

-   **DRML-Ensemble:** [https://github.com/1012lab/DRML-Ensemble](https://github.com/1012lab/DRML-Ensemble)
-   **HNNDTA:** [https://github.com/lizhj39/HNNDTA](https://github.com/lizhj39/HNNDTA)