# Predict-Customer-Personality-to-Boost-Marketing-Campaign-using-Machine-Learning



This project aims to revolutionize marketing campaigns by harnessing the power of machine learning to predict and understand customer personalities. By analyzing various data sources and employing advanced algorithms, we will develop a model that categorizes customers into distinct personality types, enabling businesses to tailor their marketing strategies to individual preferences and behavior. The project will not only increase the effectiveness of marketing efforts but also enhance customer engagement, satisfaction, and ultimately, business success.

---

## **Business Overview**

Our company's rapid growth is intricately tied to our deep understanding of customer personalities. We use historical marketing campaign data to optimize performance and precisely target potential loyal customers, driving transactions on our platform. Our key strategy involves developing a predictive clustering model, enabling data-driven decisions. By clustering customers based on behavior and personality, we provide tailored services and personalized marketing, fostering customer loyalty. Our goal is to set industry standards in customer-centric operations and sustainable growth through continuous data-driven refinement.

## **Objective 🌟**
Our primary objective is to optimize the marketing campaign by leveraging customer segmentation and data analytics. We aim to enhance customer engagement, increase conversion rates, and boost revenue while ensuring a seamless and personalized experience for our customers.

## **Goals 🎯**
Segment-Specific Targeting: Implement targeted marketing strategies for each customer segment, focusing on their specific characteristics and preferences.
Conversion Rate Optimization: Continuously refine and improve our conversion funnels, using data-driven insights to enhance the customer journey and boost conversion rates.
Customer Engagement Enhancement: Elevate the website experience and content to engage customers effectively. Develop loyalty programs and incentives to create lasting relationships.

Note: This is not a real company. The names "ShopSavvy Emporium" provided earlier are fictional names created for the purpose of this project. Please be aware that these names are entirely fictitious and not associated with any real businesses.

## **Library for The Project**

* **Pandas**
* **Numpy**
* **Scipy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**

## Data Understanding

Project Data Column Information: 

1. `Unnamed: 0`: An unnamed index or identifier column.
2. `ID`: Customer identification number or code.
3. `Year_Birth`: Year of birth of the customer.
4. `Education`: The level of education attained by the customer.
5. `Marital_Status`: Marital status of the customer.
6. `Income`: Customer's income.
7. `Kidhome`: Number of children in the household.
8. `Teenhome`: Number of teenagers in the household.
9. `Dt_Customer`: Date when the customer became a client.
10. `Recency`: Number of days since the last purchase.
11. `MntCoke`: Amount spent on Coke products.
12. `MntFruits`: Amount spent on fruit products.
13. `MntMeatProducts`: Amount spent on meat products.
14. `MntFishProducts`: Amount spent on fish products.
15. `MntSweetProducts`: Amount spent on sweet products.
16. `MntGoldProds`: Amount spent on gold products.
17. `NumDealsPurchases`: Number of purchases made with deals or discounts.
18. `NumWebPurchases`: Number of purchases made through the web.
19. `NumCatalogPurchases`: Number of purchases made from catalogs.
20. `NumStorePurchases`: Number of purchases made in physical stores.
21. `NumWebVisitsMonth`: Number of web visits per month.
22. `AcceptedCmp3`: Whether the customer accepted Campaign 3 (binary, likely a marketing campaign).
23. `AcceptedCmp4`: Whether the customer accepted Campaign 4 (binary, likely a marketing campaign).
24. `AcceptedCmp5`: Whether the customer accepted Campaign 5 (binary, likely a marketing campaign).
25. `AcceptedCmp1`: Whether the customer accepted Campaign 1 (binary, likely a marketing campaign).
26. `AcceptedCmp2`: Whether the customer accepted Campaign 2 (binary, likely a marketing campaign).
27. `Complain`: Whether the customer has registered a complaint (binary).
28. `Z_CostContact`: Cost of contacting the customer.
29. `Z_Revenue`: Revenue generated from the customer.
30. `Response`: Customer response to a marketing campaign (binary, likely indicating whether they responded positively to a campaign).


## **🚀Feature Engineering🚀**

📊 Introduction to Feature Engineering

In my quest to enhance the success of our marketing campaign, i embarked on a journey through data. Feature engineering was my guiding star, allowing me to uncover deeper insights into customer behavior and boost conversions. 💡

1. **Creating the Conversion Rate📈**
I commenced by calculating the conversion rate, a pivotal metric that measures the percentage of website visitors who responded to our campaign. This served as the foundation for comprehending customer behavior. 🧮 The calculate conversion rate are from:
**Total Responses / Total web visit**

2.   📆**Customer Age Insights**
I segmented our customers into five distinct age groups. This segmentation allowed us to gain insights into the preferences and behaviors of different age cohorts, spanning from children to senior adults. 🧑‍🦳👩‍🦳

3.   💰 **Income Labeling**
We didn't stop at numerical data; we also created a meaningful "Income Level" feature. By categorizing income into four distinct labels—Low Income, Moderate Income, High Income, and Very High Income—we gained insights into spending behavior and purchasing power. 💼💸

4.  🕰️ **Unlocking Recency Insights**
Recency is a crucial factor in customer engagement. We segmented customers based on their recency of interaction with our brand. This allowed us to tailor our marketing efforts to customers who were recently active and those who might need a gentle nudge to re-engage. 📅

5.  🛒 **Total Transactions Analysis**
Total transactions give us a comprehensive view of customer engagement. We calculated the total number of transactions for each customer, shedding light on their loyalty and engagement with our products and services. 📊🛍️

6.  👨‍👩‍👧‍👦 **Understanding Family Size**
Family size can influence purchasing decisions. We engineered the "Family_Size" feature, providing insights into the composition of our customers' households. This information is invaluable for crafting family-centric marketing campaigns. 👨‍👩‍👧‍👦🏠

7.  📆 **Recency Grouping**
To further refine our strategies, we grouped customers by recency into distinct segments. This allowed us to tailor our communication and offers based on how recently customers interacted with our brand, ensuring relevance and engagement. 📊🔍

## **Exploratory Data Analysis**

for this step we gonna investigate more about our data pattern from the distribution numerical and bar graph for categorical data.

* categorical columns = 'ID', 'Education', 'Marital_Status', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Complain', 'Response'

* numeric columns = 'Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntCoke', 'MntFruits', 'MntMeatProducts',  'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Z_CostContact', 'Z_Revenue'

### **Univariate EDA for Numeric**

1. **Checking the Data Distribution**

| Column Name         | Skewness | Kurtosis | Type of Distribution               |
|---------------------|----------|----------|-----------------------------------|
| Year_Birth          | -0.350   | 0.717    | Approximately Symmetrical (Nearly Normal) |
| Income              | 6.763    | 159.637  | Highly Positively Skewed          |
| Kidhome             | 0.635    | -0.780   | Bimodal Distribution               |
| Teenhome            | 0.407    | -0.986   | Bimodal Distribution               |
| Recency             | -0.002   | -1.202   | Approximately Symmetrical (Nearly Normal) |
| MntCoke             | 1.176    | 0.599    | Highly Positively Skewed          |
| MntFruits           | 2.102    | 4.051    | Highly Positively Skewed          |
| MntMeatProducts     | 2.083    | 5.517    | Highly Positively Skewed          |
| MntFishProducts     | 1.920    | 3.096    | Highly Positively Skewed          |
| MntSweetProducts    | 2.136    | 4.377    | Highly Positively Skewed          |
| MntGoldProds        | 1.886    | 3.552    | Highly Positively Skewed          |
| NumDealsPurchases   | 2.419    | 8.937    | Highly Positively Skewed          |
| NumWebPurchases     | 1.383    | 5.703    | Highly Positively Skewed          |
| NumCatalogPurchases | 1.881    | 8.047    | Highly Positively Skewed          |
| NumStorePurchases   | 0.702    | -0.622   | Moderately Positively Skewed     |
| NumWebVisitsMonth   | 0.208    | 1.822    | Approximately Symmetrical (Nearly Normal) |
| Z_CostContact       | 0.000    | 0.000    | Uniform Distribution              |
| Z_Revenue           | 0.000    | 0.000    | Uniform Distribution              |

#### **Histogram Visualization**

![image](https://github.com/riyouuyt/Predict-Customer-Personality-to-Boost-Marketing-Campaign-using-Machine-Learning/assets/122600889/05ecd5a8-e42b-4e67-8ad2-2a3dd54f7ad8)


#### 📊 **Summary of Distribution Characteristics**

1.  Skewness: Measures the asymmetry in data distribution.
Most columns are highly positively skewed (skewness > 1), indicating a longer tail on the right side.
Columns with skewness values between -0.5 and 0.5 are approximately symmetrical.
"Kidhome" and "Teenhome" have a bimodal distribution, suggesting two distinct modes.

2.  Kurtosis: Measures tailedness or peakedness compared to a normal distribution.
Several columns have high positive kurtosis, implying heavier tails and peaks.
Notably, "NumDealsPurchases," "NumWebPurchases," and "NumCatalogPurchases" exhibit this characteristic.

3.  Type of Distribution: Describes the distribution based on skewness values.
Most columns are highly positively skewed.
"Year_Birth," "Recency," and "NumWebVisitsMonth" are approximately symmetrical.
"Kidhome" and "Teenhome" exhibit a bimodal distribution.
"Z_CostContact" and "Z_Revenue" have a uniform distribution with constant values.
