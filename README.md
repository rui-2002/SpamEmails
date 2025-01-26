# SMS Spam Classification Project

## Overview
This project uses the **SMS Spam Collection Dataset** to train a machine learning model that classifies SMS messages as either **ham** (legitimate) or **spam**. The goal is to build a model with high accuracy for spam detection, which can be used in real-world applications like SMS filtering systems.

---

## Dataset
The dataset contains 5,574 SMS messages, each labeled as **ham** or **spam**.  
### Structure:
- **v1**: Label (`ham` or `spam`)  
- **v2**: Raw SMS message text  

Example:
```plaintext
ham   Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
spam  Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
```

---

## Requirements
To run this project, you'll need the following:

### Programming Language
- Python 3.8 or later

### Libraries

```

Key Libraries:
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn

---

## Project Structure
```plaintext
├── data/
│   ├── sms_spam_collection.csv   # Dataset (CSV format)
├── notebooks/
│   ├── EDA.ipynb                 # Exploratory Data Analysis notebook
│   ├── Model_Training.ipynb      # Model training and evaluation notebook
├── src/
│   ├── preprocess.py             # Text preprocessing scripts
│   ├── train_model.py            # Model training script
│   ├── evaluate_model.py         # Model evaluation script
├── app/
│   ├── main.py                   # Simple CLI to test the model
│   ├── spam_classifier.pkl       # Saved model file
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
```

---

## How to Run
Follow these steps to get started:

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/sms-spam-classification.git
cd sms-spam-classification
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Explore the Data
Open the **EDA.ipynb** notebook to analyze the dataset:
```bash
jupyter notebook notebooks/EDA.ipynb
```

### Step 4: Train the Model
Run the **Model_Training.ipynb** notebook or execute the training script directly:
```bash
python src/train_model.py
```

### Step 5: Test the Model
Use the CLI to test the model with custom SMS messages:
```bash
python app/main.py
```

---

## Features
- **Exploratory Data Analysis (EDA)**: Understand the distribution of ham and spam messages.
- **Text Preprocessing**: Includes tokenization, stopword removal, and stemming.
- **Machine Learning Models**: Trained using algorithms like Naive Bayes, Logistic Regression, or Random Forest.
- **Performance Metrics**: Includes accuracy, precision, recall, and F1-score.

---

## Results
The trained model achieves:
- **Accuracy**: ~98%
- **Precision**: High for spam detection
- **Recall**: Balanced for ham and spam
- **F1-score**: ~97%

---

## Acknowledgements
This project uses the [SMS Spam Collection Dataset](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/), originally compiled by Tiago A. Almeida and colleagues.

If you find this project helpful, please consider citing their paper:
Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering: New Collection and Results.* Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG'11).

---
