# SECOM Case Study: Predictive Modeling for Fault Detection

This repository contains the implementation for the SECOM Case Study, following the CRISP-DM methodology to build a machine learning pipeline for detecting defective wafers in semiconductor manufacturing. 

## Project Structure

Our repository follows a standardized structure to separate raw data, code, and outputs:

```text
secom_case_study/
├── data/
│   ├── raw/            # Local cache of the downloaded dataset (Ignored by Git)
│   └── processed/      # (Optional) Saved intermediate data states
├── notebooks/          # Jupyter notebooks for exploratory analysis and documentation
├── src/                # Reusable Python scripts (e.g., data acquisition, pipelines)
├── results/            # Exported tables, model metrics, and SIP tables
├── figures/            # Visualizations (histograms, Pareto charts, ROC curves)
├── requirements.txt    # Shared dependency list
└── README.md           # Project startup guide
```

## Getting Started

To ensure everyone is running the exact same code and package versions, please follow these steps to set up your local development environment.

### 1. Clone the Repository
Clone this shared repository to your local machine and navigate into the project directory:
```bash
git clone https://github.com/Sukant-Rajkarnikar/secom_case_study.git
cd secom-case-study
```

### 2. Create a Virtual Environment
Isolate the project's dependencies from your system-wide Python installation by creating a virtual environment:

```bash
python -m venv .venv
```


### 3. Activate the Virtual Environment
You must activate the virtual environment every time you work on this project!

**Windows Powershell Prompt:**
```powershell
.venv\Scripts\Activate.ps1
```
*(You should see `(.venv)` appear at the beginning of your terminal prompt.)*

### 4. Install Dependencies
With your environment active, install the baseline libraries required by the SECOM manual:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Running Jupyter Notebooks
```bash
jupyter notebook
```
*Note: When opening a Python file or notebook, ensure your Python Interpreter/Kernel is set to the one located inside the `.venv` folder.*

## Important Rules for Data Handling

1. **Do not commit the raw dataset to GitHub.** The data acquisition script is designed to download the data at runtime using `ucimlrepo` or `pandas` directly from the UCI ML Repository.
2. **Never touch the 20% test set.** After the initial stratified split, the test set must be locked away until the final evaluation phase. All preprocessing (e.g., missing value thresholds, outlier bounds, imputation) must be fit on the training data *only*.
3. **Log your decisions.** Update the Step Impact Protocol (SIP) table in our shared Google Docs Notebook for every major preprocessing decision (e.g., removing columns, dealing with outliers).
