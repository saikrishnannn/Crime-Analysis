India District-Wise IPC Crimes Analysis (2001–2012)
📌 Project Overview
This project performs a comprehensive spatial and temporal analysis of crimes reported under the Indian Penal Code (IPC) across various districts in India from 2001 to 2012. Using Python-based data science libraries, the project identifies crime hotspots, calculates growth rates, and implements machine learning models to forecast future crime trends.

This study was developed as a Data Science Minor Project (INT375) at Lovely Professional University.

📊 Key Features
Exploratory Data Analysis (EDA): Detailed cleaning and aggregation of NCRB district-level data.

Per-Capita Normalization: Integration of 2011 Census data to calculate "Crimes per Lakh" for fair state-wise comparison.

Anomaly Detection: Identification of statistically significant spikes in crime using Z-Score analysis.

Predictive Modeling: Comparison of three forecasting models:

Linear Regression (Baseline trend)

Polynomial Regression (Degree 2) (Capturing non-linear growth)

ARIMA (1,1,1) (Time-series forecasting with confidence intervals)

Professional Visualizations: Heatmaps, diverging bar charts, and regression dashboards using Seaborn and Matplotlib.

🛠️ Tech Stack
Language: Python 3.x

Libraries: * Pandas & NumPy (Data Manipulation)

Matplotlib & Seaborn (Data Visualization)

Scikit-Learn (Regression Models)

Statsmodels (ARIMA Time Series)

SciPy (Statistical Analysis)

📂 Dataset Source
The data is sourced from the National Crime Records Bureau (NCRB), Ministry of Home Affairs, Government of India. It includes crime heads such as Murder, Rape, Kidnapping, Robbery, Burglary, Theft, and more across 35 States/UTs.

🚀 How to Run
Clone the repository:

Bash
git clone https://github.com/saikrishnannn/Crime-Analysis.git
Install dependencies:

Bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy
Run the analysis:

Bash
python crime_analysis.py
📈 Results Summary
Highest Accuracy: The Polynomial Regression model achieved an R² score of ~0.98, significantly outperforming the standard linear model.

Trend Insight: Socio-economic crimes like "Cheating" and "Cruelty by Husband or Relatives" showed higher growth rates compared to traditional violent crimes over the 12-year period.

Normalization Impact: States with high raw crime counts (e.g., Maharashtra) show different rankings when normalized by population density (Crimes per Lakh).

👤 Author
Saikrishnan B.Tech Computer Science Engineering

Lovely Professional University
