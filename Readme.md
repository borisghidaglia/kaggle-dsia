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
    - [Year and month](#year-and-month)  
    - [Region](#region)  
    - [Orientation](#orientation)  
    - [Nature Diff](#nature-diff)  
    - [Age and age category](#age-and-age-category)  
3. [Methods](#methods)  
  3.1 [Submission 5 : Xgboost](#submission-5-xgboost)  
  3.2 [Submission 8 : Random Forest and meta groups](#submission-8-random-forest-and-meta-groups)  
  3.3 [Submission 9 : MultiOutput Random Forest stacked with a logistic regression](#submission-9-multioutput-random-forest-stacked-with-a-logistic-regression)  

## Results

Methods | Score | Date
------------ | ------------- | -------------
Xgboost | 0.60663 | 8/04
Random Forest and meta groups | 0.60018 | 9/04
MultiOutput Random Forest stacked with a logistic regression | 0.59511 | 15/04
Random Forest + Grid Search (submission_3) | 0.59327 | 27/03
Random Forest (submission_2) | 0.56932 | 26/03


## Data Exploration, Cleaning and Engineering

### Data Exploration
We will go through each interesting column, one by one, and try to find out what we could do
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

#### Year and Month
*Explanation : year and month of the case creation.*

No NaNs.

Once again, as well as the prescriptive structure plateform, those data could be
helpful to improve our score, but they will be useless for new case CRESUS will
want to predict. That is why we'll not use them.

#### Region
*Explanation : region of the buyer.*

Nans proportion :  

| Train | Test |
| :------: | :----: |
|   0.6 %   |  0.5 %  |

The largest represented region is **Île de France**, but with only **18.9%**.  
Potential solutions to complete the NaNs :
* dropping the observations with a NaN in the region column. Why not ? there are very few
of them.
* replace them by an *Unknown* value.

#### Orientation
*Explanation : the value we'll have to predict.*

No NaNs.

The cases we have in input can be classified as one of the 6 different observation
types below.

| Value | Proportion |
| :------: | :----: |
|   Surendettement   |  42 %  |
|   Accompagnement   |  40 %  |
|   Meditation   |  8.5 %  |
|   Aucune   |  8 %  |
|   Autres Procédures Collectives   |  0.9 %  |
|   Microcrédit   |  0.1 %  |

We have to pay attention to one thing : *Surendettement* and *Accompagnement*
are representing 82% of the total classifications. Thus, it could be interesting
to first classify our observations like so :  
* is it a *Surendettement* or a *Accompagnement* ?
* is it another observation ?  

And then, maybe we'll be able to classify much better into those two groups.  

One last thing : the correlation matrices I built never showed me a frank correlation
between the *Orientation* column and another one in the dataset.

#### Nature Diff
*Explanation : the reason behind the buyer situation. This information is given
by the CRESUS partners.*

Nans proportion :  

| Train | Test |
| :------: | :----: |
|   4 %   |  3.8 %  |

Overview of the possible values :

| Value | Proportion |
| :------: | :----: |
|   Endettement   |  50 %  |
|   Surendettement   |  14 %  |
|   Difficultés de gestion   |  11 %  |
|   Accident de la vie   |  8 %  |
|   Mauvaise Gestion   |  3 %  |
|   Impayés   |  3 %  |
|   Réamménagement   |  3 %  |
|   Cessation de paiement   |  2 %  |
|   etc...   |  x %  |

**Note :** there is a very small proportion of *Sur-endettement* in the data. We can
surely transform them as *Surendettement* even tho it is not very important.  

Those NaNs seems pretty hard to replace. After some digging, I wasn't able to
find a satisfying way to be almost certain that we can replace some of them by
*Endettement*. Unfortunately, even when we try to group by number of credit,
by the amount still due, etc ... There is not a distinct separation between,
*Endettement* and the other columns.

We could replace them with *Endettement*, as it is really the majority, or we
could set them as *Unknown*.

**Note :** a legitimate question is : if the *Nature Diff* value of an observation
is *Surendettement* does it means that the *Observation* value is always
*Surendettement* ? Unfortunately... no... The table below shows how the
*Surendettement* in the *Nature Diff* column is classified in the *Orientation*
column.

| Value | Proportion |
| :------: | :----: |
|   Surendettement   |  47 %  |
|   Aucune   |  23 %  |
|   Accompagnement   |  19 %  |
|   Mediation   |  11 %  |

Unfortunately, only **47%** are classified as *Surendettement* in *Orientation*...
Indeed, that is more than the average (42%, as a recall), but not significantly.

#### Age and age category
*Explanation : age of the buyer, age category of the buyer.*

Nans proportion for age :  

| Train | Test |
| :------: | :----: |
|   40 %   |  40 %  |

Nans proportion for tranche_age :  

| Train | Test |
| :------: | :----: |
|   39 %   |  39 %  |

Ok ! This percentage is different. Thus, we should be able to complete a fraction
of the age column with it.  

There are exactly 68 observations that have a *tranche age* but not an *age*.
Strangely, all of them are in the *<25* category. Well, we will hope it is not a
mistake ! The average *age* fot the *<25* category is **22.98** and the median is
**23.5**. We'll fill the missing values with **23**.

For the other NaNs, a deeper reflexion is required.  

Let's check the average and median *age* for each *situation*.

| Situation | Average | Median |
| :------: | :----: | :----: |
|   Célibataire   |  47  | 47 |
|   Concubinage   |  44  | 41 |
|   Divorce   |  56  | 56 |
|   Marié   |  53  | 52 |
|   Pacs   |  40  | 37 |
|   Veuf   |  68  | 68 |

Let's check the average and median *age* for each *profession*.

| Profession | Average | Median |
| :------: | :----: | :----: |
|   Autre   |  49  | 52 |
|   Cadre   |  47  | 47 |
|   Cadre Fonctionnaire  |  51  | 49 |
|   Chomeur  |  45  | 45 |
|   Employé  |  44  | 45 |
|   Fonctionnaire  |  47  | 47 |
|   Pro  |  50  | 51 |
|   Retraité  |  69  | 68 |

Let's check the average and median *age* for each possible value of *pers_a_charge*.

| Personnes à charge | Average | Median |
| :------: | :----: | :----: |
|   0   |  58  | 61 |
|   1   |  46  | 47 |
|   2  |  43  | 43 |
|   3  |  42  | 42 |
|   4  |  43  | 43 |
|   5  |  44  | 44 |
|   6  |  45  | 45 |
|   7  |  55  | 55 |
|   8  |  41  | 41 |
|   9  |  NaN  | NaN |


We'll use those informations to estimate the real value of the NaNs. With this
technique, we'll be able to fill the column with more correct values than if we
have done it with the global average age only.


## Methods
From the highest to the lowest score, the methods from the notebooks I decided
to submit for final score.

### Submission 5 : Xgboost

Cleaning :  
- removing 'STRUCTURE PRESCRIPTRICE', 'year', 'month' columns  
- filling NaNs that should be 0 by 0

Engineering :
- dummyfing columns
- fixing column names so that xgboost accept them
- filling age NaNs with regression + cleaning tranche_age column

Even though the titles aren't correct in the notebook, we are actually using
Xgboost for this submission. It is a pretty *naive* submission, as we only
did a little grid search.

### Submission 8 : Random Forest and meta groups

Cleaning :  
- removing 'STRUCTURE PRESCRIPTRICE', 'year', 'month' columns  
- filling NaNs that should be 0 by 0

Engineering :
- filling age NaNs with regression + cleaning tranche_age column

With this idea, we tried to merge the possible outputs into groups. We decided
to build a group with Surendettement and Accompagnement, and another one with
the other possible output. Then, On each of the two dataframe we created (in
the Surendettement and Accompagnement or not), we applied a random forest.


### Submission 9 : MultiOutput Random Forest stacked with a logistic regression

Cleaning :  
- removing 'STRUCTURE PRESCRIPTRICE', 'year', 'month' columns  
- filling NaNs that should be 0 by 0

Engineering :
- filling age NaNs with regression + cleaning tranche_age column

The idea here was to generate the probabilities that the multioutput random
forest classifier can output, and then on top of them, train another classifier:
a logistic regression. I was in a rush so I didn't cross validate (it wasn't
trivial to implement) : it was a bit risky, but I didn't seemed to overfit.
