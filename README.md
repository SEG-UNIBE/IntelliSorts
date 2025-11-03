# Presortedness‐Based Sorting Algorithm Selection

This repository presents a structured experimental workflow for investigating **sorting algorithm selection based on presortedness metrics**. The project encompasses dataset collection from Kaggle, preprocessing, training set construction, and feasibility and performance evaluations. Each Jupyter notebook represents a key stage in our research pipeline.

## Workflow and Project Structure

| Stage | Notebook | Description |
|--------|-----------|-------------|
| **1. Dataset Acquisition** | **01-DatasetCollection-KaggleDownload.ipynb** | Automates the **download of datasets** from Kaggle using the Kaggle API, ensuring reproducibility and version control. |
| **2. Data Import & Validation** | **02-DatasetCollection-KaggleImport.ipynb** | Handles **data import and validation**. Performs initial exploratory checks such as data shape, types, and missing values to ensure integrity. |
| **3. Training Set Construction** | **03-DatasetCollection-TrainingSet-D200.ipynb** | Constructs the **dataset D200**, for RQ1 and RQ2, consisting of sequences of fixed length 200. |
| | **04-DatasetCollection-TrainingSet-D400P.ipynb** | Constructs the **dataset D400+**, for RQ3 and RQ4, by selecting sequences with a minimum length of 400 elements and truncating longer ones to 10,000 elements. |
| **4. Feasibility Study (RQ1)** | **05-RQ1-FeasibilityStudy.ipynb** | Conducts a **feasibility analysis** to assess the first research question (RQ1): To what extent can a neural network interpret presortedness metrics derived from a complete sequence for accurately predicting the most efficient sorting algorithm?|
| **5. Performance Evaluation (RQ4)** | **06-RQ4-PerformanceEvaluation-FinalModel.ipynb** | Performs **final model benchmarking** to answer RQ4: Can we reduce the total number of comparisons required to sort a sequence by dynamically predicting the most efficient sorting algorithm? |
