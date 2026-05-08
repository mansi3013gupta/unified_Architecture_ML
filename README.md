# UNIFIED ARCHITECTURE FOR AUTOMATING MACHINE LEARNING

MLWiz is a user-friendly web application for performing automated machine learning (AutoML) tasks using PyCaret. The app allows you to upload datasets, perform exploratory data analysis (EDA), build various types of machine learning models, and download the trained models. It supports classification, regression, clustering, anomaly detection, and time series forecasting tasks.

## Features

- **Upload Datasets**: Upload your datasets in CSV or XLSX format.

   ![alt text](images/MLWiz.jpg)

- **Load PyCaret Datasets**: Load predefined datasets available through PyCaret. List all available datasets from PyCaret

   ![alt text](images/mlwiz1.jpeg)

- **Exploratory Data Analysis (EDA)**: Generate comprehensive EDA reports using Sweetviz, Pandas Profiling.

   ![alt text](images/mlwiz2.jpeg)

   ![alt text](images/mlwiz3.jpeg) 

- **Build ML Models**: 
  Configure preprocessing and model parameters supports
  - Feature Selection
  - Feature Transformation
  - One Hot Encoding
  - Normalization
  - Missing Data Imputation
  - Outlier Handling

  Build Machine Learning models for :

  - Classification 
  - Regression
  - Clustering 
  - Anomaly Detection 
  - Time Series Forecasting

   ![alt text](images/mlwiz4.jpeg)

   ![alt text](images/mlwiz5.jpeg)

- **Download Models**: Download the trained models for further use.

   ![alt text](images/mlwiz6.jpeg)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bitbotcoder/mlwiz.git
   cd ai-insight-hub

2. **Create and activate a virtual environment (optional but recommended):**
    ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Running the App**
    To run the Streamlit app, execute the following command in your terminal:
    ```bash
    streamlit run app.py

### Usage 
1. Choose Dataset: Select a dataset source (Upload or PyCaret) and load the dataset.
2. Perform EDA: Navigate to the "Perform EDA" section to generate an EDA report.
3. Build Model: Navigate to the "Build Model" section to configure and train a machine learning model.
4. Download Model: Navigate to the "Download Model" section to download the trained model.

### File Structure
 - `app.py`: The main entry point for the Streamlit app.
 - `ml_pipeline.py`: Contains the functions for data loading, EDA, model building, and model downloading.
 - `requirements.txt`: Lists the Python packages required to run the app.

### Dependencies
 - streamlit
 - pandas
 - sweetviz
 - pycaret

6. Install the dependencies using the command:
   ```bash
    pip install -r requirements.txt

Contributing
Contributions are welcome! 

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgements
- [Streamlit](https://github.com/streamlit/streamlit)
- [PyCaret](https://github.com/pycaret/pycaret)
- [Sweetviz](https://github.com/fbdesignpro/sweetviz)
- [Ydata profiling](https://github.com/ydataai/ydata-profiling)

📢 Share with wider community:
- [X](https://x.com/intent/tweet?hashtags=streamlit%2Cpython&text=Check%20out%20this%20awesome%20Streamlit%20app%20I%20built%0A&url=https%3A%2F%2Fautoml-wiz.streamlit.app)
- [LinkedIn](https://www.linkedin.com/sharing/share-offsite/?summary=https%3A%2F%2Fautoml-wiz.streamlit.app%20%23streamlit%20%23python&title=Check%20out%20this%20awesome%20Streamlit%20app%20I%20built%0A&url=https%3A%2F%2Fautoml-wiz.streamlit.app")
- [Facebook](https://www.facebook.com/sharer/sharer.php?kid_directed_site=0&u=https%3A%2F%2Fautoml-wiz.streamlit.app)
