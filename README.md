# **NBA Game Outcome Prediction**

This project leverages machine learning techniques to predict the outcomes of NBA games based on various features extracted from historical game data. By analyzing team performance metrics such as offensive efficiency, win percentages, and rest days, this model forecasts whether the home team will win or lose a game.

---

## **Project Overview**

The project utilizes the [NBA API](https://github.com/swar/nba_api) to extract data from an online database for NBA game analysis. The extracted data is stored in CSV format for preprocessing and feature engineering. The goal is to identify the best-performing model to predict game outcomes using machine learning techniques.

### **Features Used for Prediction:**
1. **Last Offensive Efficiency:**  
   Measures the offensive efficiency (points scored per possession) from the team's last game.

2. **Last Game Home Win Percentage:**  
   Reflects the team's win percentage in their last home game, offering insights into recent home performance.

3. **Number of Rest Days:**  
   Captures the number of rest days between the last game and the current game, indicating potential fatigue or freshness.

4. **Last Game Away Win Percentage:**  
   Represents the team's win percentage in their last away game, highlighting recent away performance.

5. **Last Game Total Win Percentage:**  
   The overall win percentage in the team's last game, providing a broad view of recent success.

6. **Last Game Rolling Scoring Margin:**  
   Calculates the rolling average of the scoring margin (points scored minus points allowed) over a specified window of games.

7. **Last Game Rolling Offensive Efficiency:**  
   A rolling average of the offensive efficiency over a specified window of games.

---

## **Goals**

The primary objective is to identify the most accurate model for predicting NBA game outcomes based on the defined features. This includes determining the optimal length of the season to train the models for improved generalization.

---

## **Project Workflow**

1. **Data Import and Cleaning:**  
   - Load the extracted CSV files.  
   - Perform preprocessing to ensure data integrity.

2. **Data Exploration:**  
   - Analyze feature distributions and relationships.

3. **Data Visualization:**  
   - Use visualizations to uncover trends and correlations in the data.

4. **Data Splitting and Preprocessing:**  
   - Split the data into training and testing sets.  
   - There are two datasets which will be used for the training and testing. These are the past season data (21-22) and the past two seasons data (20-22)
   - The cross validation dataset is the 2023 season
   - Apply scaling, encoding, and other preprocessing techniques.

5. **Model Evaluation Function:**  
   - Create a reusable function to:  
     - Train models on the training set.  
     - Evaluate performance on the testing and cross-validation sets.

6. **Model Training and Testing:**  
   - Train and test multiple machine learning models, including:  
     - Decision Tree  
     - Random Forest  
     - XGBoost  
     - Logistic Regression  
     - Naive Bayes  
     - K-Nearest Neighbors (KNN)  
   - For each model, it will be trained and tested on both the past season data (21-22) and the previous two seasons data (20-22)
   - Use classification reports to evaluate model performance on testing and cross-validation datasets.

7. **Model Comparison and Selection:**  
   - Compare performance metrics to identify the best-performing model.  
   - Determine the optimal training period for the model (e.g., full season vs. partial season).

---

## **Results**

The models were evaluated based on their testing and cross-validation weighted accuracy to determine their ability to generalize predictions for NBA game outcomes. Below is a graph and summary of the results: 

![Results Graph](image.png)

- **Logistic Regression** consistently achieved balanced performance across both testing and cross-validation datasets, with accuracies around 60%. This reflects its reliability for linear relationships in the data.
- **Random Forest and Naive Bayes** showed competitive performance, particularly for the 2021-22 data subset, achieving testing accuracies up to 64%. This highlights their ability to capture non-linear relationships and feature interactions.
- **XGBoost** demonstrated strong testing accuracy for the 2021-22 data (64%) but struggled in cross-validation, suggesting potential overfitting or sensitivity to the training subset.
- **K-Nearest Neighbors (KNN)** and **Decision Tree** models displayed moderate accuracies (56%-58%), indicating they were less effective at capturing complex patterns in the dataset compared to ensemble methods.

### **Performance Insights**
- Ensemble models like **Random Forest** and **XGBoost** provided the highest testing accuracies, confirming their strength in handling the diverse and interdependent features of NBA game data.
- Linear models such as **Logistic Regression** maintained consistent performance, offering a stable baseline for comparison.
- The performance gap between testing and cross-validation accuracy for certain models (e.g., XGBoost) emphasizes the importance of addressing overfitting and fine-tuning hyperparameters.

The graph below provides a visual representation of these results, highlighting each model's performance across different data subsets and evaluation metrics.

---

## **Technologies Used**

- **Python:** Data preprocessing, modeling, and evaluation.  
- **NBA API:** Data extraction and feature engineering.  
- **Libraries:**  
  - `scikit-learn` for machine learning models.  
  - `pandas` and `numpy` for data manipulation.  
  - `matplotlib` and `seaborn` for data visualization.

---

## **Future Work**

- Fine-tuning hyperparameters for ensemble models like Random Forest and XGBoost.  
- Experimenting with additional features or engineered variables.  
- Expanding the analysis to include overtime or playoff game predictions.

