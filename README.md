# E Commerce Customer Churn Predictor

Analysing the purchase history  of customers to predict which customer is likely to stop buying.

---

## Project Overview

In retail customers don't send a message when they leave—they just stop buying. This project uses a dataset of over 500 000 transaction records to:

1. Turn store receipts into customer profiles.

2. Mark any customer who hasn't bought anything in 90 days as "Churned".

3. Train a Machine Learning model (Random Forest) to spot at-risk customers early so the marketing team can send them offers or reminders.

---

## How It Works

### 1. Cleaning the Data

- Removing inconsistent data like cancelled orders, negative item counts and missing customer IDs.

- Grouped purchases by customer:

Recency: How many days ago they last bought something.

Frequency: Total orders they placed.

Monetary: Total money they spent.

### 2. Preparing Features & Labels

- Set a rule: If Recency < 90 
		then, 
			label customer as churned (1)
		otherwise,
			label = active (0)

- Created a metric called Average Order Value (Total Spend / Total Orders):

	to help the model tell the difference between big spenders and casual buyers.


### 3. Model. Challenges

- The dataset had active users (~66%) than churned users (~33%).

- This imbalance makes it harder for the starting model to find churned users.

- Setting class weights to "balanced" forced the model to give importance to churned users, which helped it catch at-risk customers much better.

---

## Data Visualization

- Created simple scatter. Pie charts to look at customer purchase habits.

- Used clean basic color schemes so the charts are easy to read and understand.

---

## Key Challenges & Lessons Learned

1. **Handling Class Imbalance**
   The raw dataset was heavily weighted toward active users (~66%) compared to churned users (~33%). Because of this imbalance, the baseline Random Forest model initially gave a low recall score of 0.43 for churned customers, as it favored predicting the majority class. 
Setting `class_weight='balanced'` in the model parameters forced the algorithm to pay equal attention to churned users, significantly improving its ability to catch at-risk customers.

2. **Feature Engineering**
   Raw spend and order counts didn't give the full picture on their own. Adding `AvgOrderValue` (Total Spend / Total Orders) helps the model distinguish between high-volume, repeat buyers and single-item, one-off purchasers.

3. **Data Visualization**
   The initial scatter plots were crowded and hard to read because hundreds of customer data points overlapped. Switching to box plots and using neutral, high-contrast color schemes made it much easier to compare spending habits across active and churned groups.
---

## Making Predictions for New Customers

The notebook includes a Python function called predict_customer_churn(). You can pass in an order count and total spend, for any customer. It returns their risk status and churn probability:

```python

# Example:

predict_customer_churn(frequency=1, total_monetary=15)

# Returns:

# Status: CHURN RISK

# Probability: 92.0%
```

---

## 🛠️ Tech Stack & Tools
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn (Random Forest)
