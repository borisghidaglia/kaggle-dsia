# Indebtedness Case Orientation

[CRESUS association](https://www.cresus-iledefrance.org/) advise people that are
in a indebtedness situation or that are facing difficulties managing their budget.
Given a history of situations, the project is about suggesting the best orientation
possible for a new situation.

**Note : the entire competition took place in french and the data is in french
too. In this Readme, I translated everything you could need to understand the
competition, but CRESUS is a french association thus the original data will
remain in french.**

Link to the competition : [Indebtedness Case Orientation](https://www.kaggle.com/c/dsia-printemps-2019)

## Table of contents

1. [Results](#results)  
2. [Data Exploration, Cleaning and Engineering](#data-exploration-cleaning-and-engineering)  
  2.1 [Data Exploration](#data-exploration)  
    - [NaNs overview](#nans-overview)    
    - [Prescriptive Structure](#prescriptive-structure)  
    - [Platform](#platform)  


## Results

Methods | Score | Date
------------ | ------------- | -------------
Random Forest + Grid Search (submission_3) | 0.59327 | 27/03
Random Forest (submission_2) | 0.56932 | 26/03


## Data Exploration, Cleaning and Engineering

### Data Exploration
We will go through each column, one by one, and try to find out what we could do
with each one. But first, some general informations.  

#### NaNs overview
Let's concatenate test and train, to have a more general point of view of the
data. Then, we begin by checking the NaN proportion, by column.  

```
gain_mediation             0.972847
crd_decouvert              0.955671
crd_rac                    0.933822
crd_autres                 0.882546
cat_impayes                0.843016
IMPAYES_DEBUT              0.843016
crd_immo                   0.824577
age                        0.400606
tranche_age                0.392018
crd_renouv                 0.280121
crd_amort                  0.206365
PROF                       0.101667
NATURE_DIFF                0.041803
moy_eco_jour               0.023112
cat_moy_eco_jour           0.023112
RAV_UC                     0.022354
situation                  0.022354
cat_RAV_UC                 0.022354
adulte_foyer               0.022354
LOGEMENT                   0.019955
region                     0.006188
cat_credit                 0.000126
pers_a_charge              0.000000
PLATEFORME                 0.000000
year                       0.000000
month                      0.000000
nb_decouvert               0.000000
nb_autres                  0.000000
nb_rac                     0.000000
nb_immo                    0.000000
nb_credits                 0.000000
nb_renouv                  0.000000
REVENUS                    0.000000
cat_rev                    0.000000
CHARGES                    0.000000
cat_charges                0.000000
CREDITS                    0.000000
nb_amort                   0.000000
CRD                        0.000000
RAV_ouverture              0.000000
cat_RAV_ouverture          0.000000
STRUCTURE PRESCRIPTRICE    0.000000
```
Well, I think we have some work to do ! Almost half of the features contains NaNs.
We better start right now.

#### Prescriptive Structure
*Explanation : Reference of the structure from which the case is comming.*

No NaNs.

We noticed that some structures in the train dataset are missing in the test
dataset and vice versa.  

First of all, I thought that a prescriptive structure could be more likely to
give more case for this or that orientation. Indeed, if for example one of them
is located in a very poor area, none of its case will be oriented as "rejected"
or equivalent.  

But then, I figured that the association will never need to classify an old case.
Thus, even if this feature could improve our prediction score, using it seems to
be a nonsense, or could even lead to some bad predictions in the future, when
the prescriptive structure will have change.

#### Platform
*Explanation : Whether it is the banking or social plateform of CRESUS.*

| Bancaire | Social |
| :------: | :----: |
|   72 %   |  28 %  |

No NaNs.

Similarly spread between the possible orientations : no obvious insight here.
**Note :** amongst the observations, no "microcredits" has been chosen by the
social plateform of CRESUS - but there are very few microcredits, thus it is not
very interesting.


## Methods

### Random Forest


### Random Forest and Grid Search
