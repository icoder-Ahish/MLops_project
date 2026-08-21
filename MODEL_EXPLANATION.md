# End-User Guide: Understanding the Stock Price Prediction Model

---

## 1. What does this Model do? (Simple Explanation)

Imagine you are watching a stock's daily closing prices. You want to know what tomorrow's price might be. 

This model acts like a **smart price assistant**:
- It looks at **5 consecutive days of stock prices** (the "Lookback Window").
- It identifies trends (e.g., rising, falling, or fluctuating).
- It **predicts the stock closing price for the next day (Day 6)**.

---

## 2. Real-World Example

Suppose you feed the model the stock prices of Company XYZ for the last 5 trading days:

| Day | Date | Closing Price ($) |
|---|---|---|
| Day 1 | Monday | $150.25 |
| Day 2 | Tuesday | $152.10 |
| Day 3 | Wednesday | $149.80 |
| Day 4 | Thursday | $153.45 |
| Day 5 | Friday | $151.90 |

**What the Model Does:**
The model processes these 5 numbers and outputs **1 predicted number**:
> **Predicted Closing Price for Day 6 (Next Monday):** `$154.20`

---

## 3. What Happens When You Give It 10 Days of Data?

If you provide 10 days of stock prices instead of 5, the model uses a **sliding window** of 5 days at a time:

```
Day 1 to 5  ───> Predicts Day 6  ($154.20)
Day 2 to 6  ───> Predicts Day 7  ($156.10)
Day 3 to 7  ───> Predicts Day 8  ($155.00)
Day 4 to 8  ───> Predicts Day 9  ($157.50)
Day 5 to 9  ───> Predicts Day 10 ($158.80)
Day 6 to 10 ───> Predicts Day 11 ($160.10)
```

That is why giving 10 historical prices results in **6 predicted values**!

---

## 4. Under the Hood (How it Works Step-by-Step)

1. **Normalizing (Scaling):** Stock prices can be $10, $150, or $3000. The model standardizes all prices into numbers between 0 and 1 so it can recognize pattern shapes rather than raw values.
2. **Pattern Recognition (LSTM Neural Network):** An **LSTM (Long Short-Term Memory)** network reads the 5-day sequence to learn the short-term market momentum.
3. **Unscaling (De-normalizing):** The prediction (between 0 and 1) is converted back into actual dollar/rupee stock values.

---

## 5. Summary Table

| Question | Answer |
|---|---|
| **What goes in?** | Minimum 5 past days of stock closing prices |
| **What comes out?** | Predicted stock closing price for the next day |
| **Who is it for?** | Traders/Analysts wanting AI-driven price trend forecasts |
| **Can I input 1 price?** | No, minimum 5 prices are required to detect a pattern |
