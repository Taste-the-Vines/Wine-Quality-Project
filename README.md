# Wine-Quality-Project

## Project Description
#### Data pulled from Data.world. The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. Number of wines representing datapoints: red wine - 1599; white wine - 4898. 11 features were utilized (reference: (https://data.world/food/wine-quality)).

## Goals 
  * Analyze wine composition data and the 3-9 quality classification
  * Establish a baseline classification for wine target value
  * Generate a model that beats the baseline in acuracy

## Initial Hypothesis
###

## Project Plan
 * Acquisition
   * two datasets on Red and White wine pulled from Data.world
 * Preparation 
   * two data sets merged 
   * checked for nulls (there were none)
   * created column for color of wine (see data dictionary)
   * outliers removed
   * data split into Train, Validate, and Test sets
 * Exploration
   * vizualizaitons 
   * bivariate stats analysis
   * see below
* Feature Engineering
  * dummies created
  * clustering experimented with (see below)
##  Exploration
### Exploration Question 1:
* Is the average alcohol content in low quality wine significantly lower than the alcohol content in high quality wine?
  -  𝐻0
   : There is no difference the average alcohol content between wines with a quality of less than 5 and wines with a quality greater than 7
  -  𝐻𝑎
   : The average alcohol content of wines with a quality less than 6 is significantlly lower than the average alcohol content of wines with a quality greater than 6
### Exploration Question 2:
* Is the average chloride in high quality wine significantly lower than the average chloride for all wines?
  -  𝐻0
   : There is no difference the average chloride in wines with a quality greater than 6 and the average chloride in all wines
  -  𝐻𝑎
   : The average chloride of wines with a quality greater than 6 is significantlly lower than the average chloride of all wines
### Exploration Question 3:
* Is the average citric acid in low quality wine significantly lower than the average citric acid in high quality wines?
-  𝐻0
 : There is no difference the average citric acid in wines with quality lower than 6 and the average citric acid in wines with quality greater than 6
-  𝐻𝑎
 : The average citric acid of wines with quality lower than 6 is significantlly lower than the average citric acid of wines with quality greater than 6
 ### Exploration Question 4:
*Is the average pH in low quality wine significantly lower than the average pH in high quality wines?
-  𝐻0
 : There is no difference the average pH in wines with quality lower than 6 and the average pH in wines with quality greater than 6
-  𝐻𝑎
 : The average pH of wines with quality lower than 6 is significantlly lower than the average pH of wines with quality greater than 6
#### * see explore.py module for notations on exploration functions
## Preparation
### * see prepare.py for notations and information on further disposition of data during and after acquisition
## Modeling
  * Models used for Train and Validation
  * Random Forest iterations (best selected)


## Data Dictionary
| Feature | Description |
| ------ | ----|
| fixed acidity | most acids involved with wine or fixed or nonvolatile (do not evaporate readily) (tartaric acid - g / dm^3)   |
| volatile acidity | the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste (acetic acid - g / dm^3) |
| citric acid | found in small quantities, citric acid can add ‘freshness’ and flavor to wines (g / dm^3) |
 | residual sugar | the amount of sugar remaining after fermentation stops, it’s rare to find wines with less than 1 gram/liter and wines with greater than 45 grams/liter are considered sweet (g / dm^3) |
| chlorides | the amount of salt in the wine (sodium chloride - g / dm^3) |
| free sulfur dioxide | the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents microbial growth and the oxidation of wine (mg / dm^3) |
| total sulfur dioxide | amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2 concentrations over 50 ppm, SO2 becomes evident in the nose and taste of wine (mg / dm^3) |
| density | the density of water is close to that of water depending on the percent alcohol and sugar content (g / cm^3) |
| pH | describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale |
| sulphates | a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial and antioxidant (potassium sulphate - g / dm3) |
| alcohol | the percent alcohol content of the wine (% by volume) |
| color | color of the wine (red or white) |
| quality | rated by sensory observation of expert (score between 0 and 10) TARGET VALUE |
 ### reference   https://rstudio-pubs-static.s3.amazonaws.com/57835_c4ace81da9dc45438ad0c286bcbb4224.html 


## Steps to Reproduce
1. Clone this repo
2. Obtain the data from Data.world and use the fuctions from prepare.py module.
3. Run the explore and modeling notebooks
4. Run final report notebook using the explore.py and modeling.py
