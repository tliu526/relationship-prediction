# Relationship prediction journal

## Lessons learned

- Workflow that works:
  - prototype test code in jupyter notebook
  - write tests and port over into actual unit tests
  - document as you go
  - clean jupyter notebook as final outputs
  - write data processing scripts with argparse
  
  - unit test with small slices of data -> synthetic is best

- What didn't work:
  - "thrashing" on experiments: make sure to have a clear direction when conducting runs, else you waste time
  - shell scripts: need a better way to track the entire data transformation pipeline, up to training
    - MakeFile?

## 2019-03-07

### Paper discussion

- There is a dichotomy in that demographics only help significantly when we're classifying _all_ contacts, the effects go away when we only consider the top 5

### Notes

- could we sort contacts based on the demographic information itself -- and remove that feature?
  - allows us to categorize more precisely who lives together and who lives apart
- can target WellComp 2019 (late July deadline) for extensions of the wellbeing prediction

## 2019-03-05

### Notes

- make sure to clearly define any terminology utilized in the paper
- qualification: age has a large effect only when considering _all_ contacts, when we limit to top 5 the feature importance becomes less prominent
- try running top5_all, top5_demo automl with more memory, more time

### Top 5 performance tables

rows: models, including majority baseline
columns: acc, balanced acc, weighted f1, macro f1, precision, recall

## 2019-03-04

### New results outline

- top 5 relationship prediction (six classes)
  - make sure to discuss why we chose to collapse the classes
  - feature importance:
    - SHAP
    - point biserial correlations
- Life facet replication
  - state our limitations, what we've done differently
  - feature importance
    - SHAP 
    - point biserial correlations
- Tie strength replication
  - state our limitations
  - feature importance
  
### Notes

- tie str nan dedup is the macro f1 run


## 2019-03-03

### Notes

- it appears that the additional data dumps still do not have correct GPS coordinates, perhaps they are intentionally obfuscated
- PDF checker for MobiCom submission: https://www.sysnet.ucsd.edu/sigops/banal/index.php
- ACM conference proceedings template: https://www.overleaf.com/latex/templates/acm-conference-proceedings-new-master-template/pnrfvrrdbfwt
- top 5 Zimmerman run?

- paper experiments:
  - top 6 classification
  - top 4 classification (???)
  - top 5 Zimmerman
  - Zimmerman all
  - tie strength

- insights:
  - effect of demographics on predictive power
  - predicting closest relationships vs predicting all relationships fundamentally different tasks
  - highlights need for considering population heterogeneity for phone studies

### Scripts ran

- Note: final runs are nan dedup'd, and 6 time of day divisions instead of four

```bash
# for testing whether the final location features actually help
python build_features.py --comm_file ../data/top5_final.df ../data/final_sandbox/top5_all --emc_features --impute_missing --impute_median --loc_features --dedup_nan

# for double checking Zimmerman replication results
python build_features.py --comm_file ../data/zimmerman_contacts.df ../data/final_sandbox/zimmerman_demo --emc_features --impute_missing --impute_median --demo_features --dedup_nan
```

## 2019-02-25

### Considering Population Age Variation in Phone-Based Relationship Prediction

Estimating the nature and quality of interpersonal relationships from phone data could be useful for studies of mental well-being and social support. Prior works utilized the volume of communications to estimate broad relationship categories. In this paper, we contextualize communication events by combining phone logs with demographic and location data to predict social roles as well as relationship qualities such as closeness. We obtain the most benefit in model performance by including participant age, which is not an additive factor but rather an interaction with communication trends across social roles. Our findings not only illustrate the value of utilizing data across different modalities, but also underscore the importance of considering population heterogeneity in phone-based mental health studies.

## 2019-02-24

### To-dos

- analyze tie strength runs, where is age in that feature importance?
- write an abstract
- validate bumps in accuracy with 10-fold
- look at show_models(), cv_results_ in auto-sklearn model

### Notes

- 3% bump in test accuracy across all contact prediction
- 1.5% bump in top 10 contact prediction
- 
- potential story for tie strength:
  - clearly, no variation in tie strength as a function of age across categories
  - and yet, including age as a feature helps, significantly. Why? -> age as a multiplicative factor

## 2019-02-23

### Goal for the weekend

- create "talk outline" figures
- Considering Population Heterogeneity in Personal Sensing Relationship Prediction

### Experiments to run

- {with other / no other} {baseline, age_gender, demo} macro f1 tie strength dedup nan (6 experiments)
- {baseline, age_gender, demo} weighted f1 all contacts 4clf dedup nan (3 experiments) (dolores)
- figure out what's going on with 10-fold CV

  
## 2019-02-22

### Meeting with Sharath

- demographics analysis
- correlation table with age and communication features
- visualizations of population heterogeneity -> look at SHAP for automl visualization

### Experiments to run

- weighted/macro F1 age/gender replication
- non-group CV for Zimmerman replication
- top 10 contact type prediction?
- need something that shows age feature importance -> LIME/SHAP
- tie strength prediction with macro F1 and demographic information
  
## 2019-02-19

### Plan

- build precision/recall/f1 tables (make them look nice), update in jupyter notebooks
- add feature importances to Overleaf
- write discussion

- hard stop after Architecture to send update

## 2019-02-17

### Experiments ran

- 2 class baseline tie str classification for svm, automl on dolores (~6 pm completion time)
- LOOCV 
- try dumping the nan indicators


## 2019-02-16

### Plan for tomorrow

- feature importance with random forest
- collate final replication results
- figure out LOOCV with automl.refit
- figure out 10-fold CV with automl.refit
- combine into final notebook
- email Sharath

- EDA for blood pressure
- read JNC8 paper, look for references

### Experiments ran

- downsampled baseline and (with new loc) all Min et al replication (quadcorn)
- downsampled **original** all Min et al replication (dolores) (ETA 7pm completion)
- tie str rank classification with baseline, all, and orig all (SVM and AutoML) (ETA 7:42 completion)
- tie str score regression with baseline, all, and orig all (RF and AutoML) (10:30 pm kick off)

## 2019-02-14

### Final final final experiments

- top 5 prediction on the models trained on the entire dataset
- properly CV'd life facet prediction
- tie strength replication
- additional location features
  - outgoing call/text per location
  - call duration per location
  - different normalization(?)
- single score summary (trust score, support score possibilities)

- finish writing paper over the weekend, communicate with Sharath the updates
- one vs many classifier? -> look into AutoML implementation

## 2019-02-13

### Meeting outline

- discuss Aldo paper
- find a threshold optimization paper on blood pressure
- AutoML paper update
  - meeting with Sharath Wednesday
  - deadline of end of this week to finish

## 2019-02-10

### Plan for Wednesday

- finish writing methods
- draft introduction
- draft discussion
- look for visualization of statistics that is compelling

### Implementation clean-up

- document Github repository
- create experiment scripts
- implement proper cross validation for classification

### Additional experiments

- replicate tie strength predictions
- location-only feature importance

### Notes

- kicked off 24 hours training for relationship closeness on dolores, baseline and all features

## 2019-02-07

## Random idea extensions

- classify "other" contact types
- use the "all" model to predict top 5
- try an "all-day" run


## 2019-02-06

### Meeting notes

- David: frame the story in terms of a replication problem
- the "interpretation" results aren't super compelling
- R values are acceptable for psychology, but not useful from a practical perspective

### Action Items

- ~~email Jun-Ki~~
- ~~rewrite abstract~~
- look back to the Ben Recht paper for replication context

## 2019-02-05

### Meeting notes

- conferences/journals to submit to
- resampling method: split into group k-folds and then resample each fold

- plan for tomorrow
  - show current results for Zimmerman replication
  - resample PROPERLY and try the collapsed task
  - look at score prediction
- 8 runs for tonight:
  - collapsed classes all features
  - baseline regression for closeness
  - all regression for closeness

### To-Dos

- start putting methods section together
- start putting introduction section together
- put together jupyter slides for Zimmerman replication
  - their resampling strategy
  - their data distribution
  - look up statistical significance testing for accuracy results
- put together figures for the final, final results

### Experiments

- ~~baseline/all AutoML for Zimmerman labels~~
- AutoML for normal labels:
  - ~~baseline~~ (rerunning because of mem issues)
  - age/sex
  - demo
  - loc
  - ~~all~~ (rerunning because of mem issues)
- AutoML for collapsed labels:
  - ~~baseline~~ (trying SMOTE vs Random over sampling)
  - age/sex
  - demo
  - loc
  - all
- RF for normal labels:
  - baseline
  - age/sex
  - demo
  - loc
  - all
- RF for collapsed labels:
  - baseline
  - age/sex
  - demo
  - loc
  - all
- SVC for Zimmerman labels:
  - ~~baseline~~
  - ~~all~~

## 2019-02-03

### Goals for today

- process all Min features for exact replication

### Min experiment parameters

- random resampling for class imbalance

## 2019-02-01

### Goals for today

- process all the Min features, for exact replication

### Missing features

- ~~total duration calls~~
- ~~total \# lengthy calls (double the average/median duration)~~
- ~~convert avg, std, min, max, med \# calls/text per week~~
- ~~duration calls at time of day, days of week / total calls~~
- ~~\#{lengthy, failed, missed} calls at time of day/ days of week~~
- ~~{\#,  dur} calls for the past two weeks / total calls~~
- ~~\# outgoing comms / total comms~~
- \# comms at times of day, days of a week / total comms
- ~~\# calls / total COMMs at times of a day, days of a week~~
- ~~outgoing COMMS at {times of a day, days of a week, holidays} / total COMMs~~
  - ~~holidays are Christmas, New Year's, Valentine's Day, Thanksgiving~~
- ~~\# COMMs for the {past two weeks/ past six weeks} / total COMMs~~

## 2019-01-28

### Goals for today 

- begin putting together notebook story
- look at Max's features
- investigate training auto-sklearn with weighted F1 as targeted metric
- run autoML random forest baseline results
- make "hard task" training runs in the evening
- TODO fix git idrsa for quadcorn

### Tasks accomplished

# 2019-01-27

### Tasks accomplished

- implemented call duration feature
- implemented min/med/max feature

## 2019-01-25

### Meeting notes

- look at stratified CV for sampling perhaps
  - to investigate why adding gender makes things worse
- trait vs state (long term vs short)
- take those question results as outcomes
- write a talk as an outline
- look at categoricalized EMC
- abstract should state 3 values, perhaps: RF F1, baseline F1, all features F1

call  <-> relationship type <-> questionnaire <->

- look at correlation between relationship type and questionnaire
  - can explain/justify the confusion matrices of the classifications


## 2019-01-24

### Ideas

- two-stage prediction, use predicted EMA responses
- feature importance one vs all: random forest
- look for a feature interpretation story

### Deadlines

- MobiCom: March 8th
- ACM IMWUT: 2/15, 5/15
- JMIR (?)
- CHI: computer human interaction

### Meeting agenda

- solidifying a story
  - prediction bump: re-run experiments?
    - resampling
    - more features
  - feature importance
- figuring out a timeline
- Potential story: obviously if you ask questions about a person's relationship with someone that is predictive of their role, but our passive features approximate the predictive power of these questions

### Additional threads

## 2019-01-23

- continue driving the paper forward
- look at log loss as the cost functions
- look at class imbalance

## 2019-01-22

- found bugs in top_10 location generation (top 5 is fine)

### Things to run

- random forest baselines
- top 10 all
- top 10 loc (in progress)

###

1. base communication only
2. comm + age/sex
3. comm + demo
4. comm + loc
5. everything


## Meeting topics

- resampling?
- 

## 2019-01-21 

### Experiments completed

- top 5, top 10, baseline comm features
- top 10 comm + age/gender features
- top 10 comm + demographic features
- ~~top 10 comm + location features~~

## 2019-01-20

### Experiments to run

- top 5 and top 10 contact prediction
- Five separate prediction tasks:
  - `contact_type`: classification
  - `q1_want`: regression
  - `q2_talk`: regression
  - `q3_loan`: regression
  - `q4_closeness`: regression

#### Experiments

- baseline random forest
- communication features only
  - Konrad's additional feature engineering
  - demographic features: age/sex of ego
  - semantic location features
  - additional demographic features from survey
  - mood disorder features?

## 2019-01-16 Meeting notes

### What I want to get out of the meeting

- ~~additional models to train, feature variation~~
- figure design
- ~~what would make the argument more compelling?~~
- ~~how to integrate mood disorder analysis? Is this pigeonholing?~~

### New abstract: measuring social relationships and their meaning using phone data 

#### Context

- relationship estimation
- trust, social support

Being able to estimate the strength and quality of interpersonal relationships could be useful for studies of mental wellbeing and social support groups.

#### Content: gap statement

- AutoML to defend against overfitting
- addition of location, demographic context
- prediction of contact-based EMAs
- I would talk to this person about important matters
- I would be willing to ask this person for a loan of $100 or more
- How close are you to this contact?

Past approaches aimed at solving this problem focused on the volume of call and SMS data. However, relationship information is often revealed by contextualizing these communications: the recipient of a call placed at 6pm could be vastly different if the caller is at a bar or at home, or whether the caller 18 years old or 58 years old. Here we utilize passive phone sensor data collected from 200 participants across communication, location, and demographic modalities build models that predict the social roles of participant's contacts as well as qualitative aspects of their relationship such as trust and closeness. To safeguard against engineering bias and overfitting, we apply automatic machine learning methods to these prediction tasks. We find that the inclusion of location and demographic features greatly improves performance over communication data alone, which suggests that future relationship quality studies can be enhanced by collecting data in these modalities.

#### Conclusion

- can better inform contact labelling
- can better inform what passive features to collect from these studies for relationship prediction
- gives insight into some psychological aspects of user support
  - asymmetric communication

#### Feature blocks

- basic demographic information
- "smarter" communication features
- location-based features