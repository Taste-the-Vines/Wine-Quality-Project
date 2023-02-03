# Wine-Quality-Project

## Project Description
#### Data pulled from Data.world. The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. Number of wines representing datapoints: red wine - 1599; white wine - 4898. 11 features were utilized (reference: data.world/food/wine/wine-quality).

## Goals 
#### - Analyze wine composition data and the 3-9 quality classification
#### - Establish a baseline classification for wine target value
#### - Generate a model that beats the baseline in acuracy

## Project Plan
### - Acquisition
#### two datasets on Red and White wine pulled from Data.world
### -Preparation 
#### two data sets merged 
#### checked for nulls (there were none)
#### created column for color of wine (see data dictionary)
#### outliers removed
#### data split into Train, Validate, and Test sets
### -Exploration
#### vizualizaitons 
#### bivariate stats analysis
#### see below
### -Feature Engineering
#### dummies created
#### clustering experimented with (see below)
## Exploration
### -
### -
### -
### -
#### * see explore.py module for notations on exploration functions
## Preparation
### * see prepare.py for notations and information on further disposition of data during and after acquisition
## Modeling
### Models used for Train and Validation
#### Random Forest iterations (best selected)

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
|reference | https://rstudio-pubs-static.s3.amazonaws.com/57835_c4ace81da9dc45438ad0c286bcbb4224.html |



