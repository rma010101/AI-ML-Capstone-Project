# Predictive Modelling of HDB Resale Prices

## Overview
This project focuses on the application of machine learning to predict HDB resale prices and providing valuable insights for various stakeholders. It highlights the 
dual goals of developing a predictive model and offering decision support, making it clear and comprehensive.

## Data Source
This dataset includes resale price information for HDB flats in Singapore, covering the period from January 2017 to June 2024. It contains records from reliable sources such as the URA, and other relevant databases. The dataset is comprehensive, showing details such as transaction month, town, flat type, block, street name, storey range, floor area, flat model, lease commencement date, remaining lease, and resale price.

For this project, the dataset was downloaded from Kaggle using Python code, which originally sourced it from data.gov.sg. 

[Dataset](Data/sg-resale-flat-prices-2017-onwards.csv)

![Data Info](Data/hdb_df_info.png)

The dataset contains 181,262 entries and 11 feature columns. These columns are: 'month', 'town', 'flat_type', 'block', 'street_name', 'storey_range', 'floor_area_sqm', 'flat_model', 'lease_commence_date', 'remaining_lease', and 'resale_price'.

All feature columns, except for three, contains categorical datatypes. The features 'floor_area_sqm' and 'resale_price' are of the float datatype, while the feature 'lease_commence_date' is of the integer data type.

Importantly, the entries to all the features are complete ie there are no null entries or missing values. 

#### Feature Removal
In the initial stages of this data preparation, features deemed inconsequential to the resale price were removed. The selection was based on a judgement call, leveraging my familiarity and understanding of the housing market. Later, a correlation analysis will be conducted to determine which of the remaining features significantly impact the resale price. 

The features removed were ‘street_name’, ‘flat_model’ and ‘block’ (refer to figure below).

![Initial Features Removed](Data/hdb_df_feature_removed_001.png)

#### Determining Outliers
Of the feature columns available, 2 features ie 'floor_area_sqm' and 'resale_price' contained numerical data types and are candidates for outlier analysis.

The boxplot below for ‘floor_area_sqm’ indicates significant outliers, with many entries exceeding 160 square meters and falling below 45 square meters. Upon reviewing the smallest and largest floor areas for HDB flats in Singapore, which range from 45 to 186 square meters, it is reasonable to exclude entries below 45 square meters and above 186 square meters. This adjustment will help ensure the dataset accurately represents typical HDB flat sizes. 
