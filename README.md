# Mental Health Predictor

## Overview

Mental Health Predictor is a web application designed to predict whether a person is at risk of depression based on various personal and lifestyle factors. The app uses a machine learning model trained on over 140,000+ data points to assess mental health and provides insights based on user input.

This app is deployed on Streamlit Community Cloud. A live version of the application can be found [here](https://mentalpredict.streamlit.app/).

## Features

- **Depression Risk Prediction** : The app predicts whether a user is at risk of depression based on input data such as age, sleep duration, work pressure, etc.
- **Interactive Input** : Users can adjust input values like age, work pressure, family history, and more using interactive sliders and selectboxes.
- **Probability Display** : The app displays the probability of depression based on the user’s inputs.
- **Visualization** : Users can visualize how different factors affect mental health through interactive charts.

## Requirements
To run this app, you'll need the following Python packages (refer requirements.txt): 

```bash
streamlit
pandas
numpy
pickle
matplotlib
seaborn
category-encoders (for encoding categorical variables)
scikit-learn (for model training)
```
## Installation

1. Clone this repository or download the files.

2. Install the required libraries by running:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## How It Works
- Data Input: Users are asked to provide details such as their age, profession, sleep hours, and work pressure using interactive sliders and dropdown menus.
  
- Prediction Model: The app uses a trained machine learning model (Random Forest Classifier) to predict the likelihood of depression based on the input data.

- Output: The app shows whether the user is at risk of depression and provides the probability score.

- Visualization: The app visualizes the relationship between input factors and mental health using interactive plots.

## Screenshots
![Screenshot 2024-12-31 125735](https://github.com/user-attachments/assets/abfcdebd-1a85-407b-8cf3-6e3cd5b1678d)
![Screenshot 2024-12-31 125759](https://github.com/user-attachments/assets/f7cb253f-3e4a-416a-805a-638473b4dd9c)
![Screenshot 2024-12-31 125821](https://github.com/user-attachments/assets/f25ca0b2-cdea-4afe-9e61-e51e922e53a6)



