# Overview dsend_project_2:
- [Repository content](#Repository_content)
- [Software requirements](#Software_requirements)
- [Motivation](#Motivation)
- [Task](#Task)
- [How to run](#How_to_run)
- [Summary of results](#Summary_of_results)
- [Acknowledgements & licensing](#Acknowledgements)

## Repository content:<a name="Repository_content"></a>
This repository includes .csv, .db, .html, .pkl, and .py files for the second project of the Udacity Data Scientist for Enterprise Nanodegree.

```bash
.
├── data
│   ├── DisasterResponse.db
│   ├── disaster_categories.csv
│   └── disaster_messages.csv
├── models
│   └── classifier.pkl
├── process_data.py
├── train_classifier.py
└── webapp
    ├── run.py
    └── templates
        ├── go.html
        └── master.html
```

## Software requirements:<a name="Software_requirements"></a>
Python version 3.x & the followings packages:
- numpy==1.18.1
- pandas==1.0.1
- sklearn==0.22
- SQLAlchemy==1.3.13

## Motivation:<a name="Motivation"></a>
Twitter allows its users to share concerns with the public. In the case of an emergency, making others aware of a problem can potentially save lifes. When time is short, however, it could turn out to be challenging to interpret many Twitter messages quickly.

The idea behind this project is thus to analyze and to categorize Twitter messages. Thereby rescuers can understand faster how to help people in need. This idea has been originally suggested by Figure Eight Inc.

## Task:<a name="Task"></a>
"Multilabel Classification [...] This can be thought of as predicting properties of a sample that are not mutually exclusive."(https://scikit-learn.org/stable/modules/multiclass.html)

"multilabel data is specified as an indicator matrix, in which cell [i, j] has value 1 if sample i has label j and value 0 otherwise." (https://scikit-learn.org/stable/modules/model_evaluation.html#average)

"In multiclass and multilabel classification task, the notions of precision, recall, and F-measures can be applied to each label independently. There are a few ways to combine results across labels, specified by the average argument to the average_precision_score (multilabel only), f1_score, fbeta_score, precision_recall_fscore_support, precision_score and recall_score functions, as described above. Note that if all labels are included, “micro”-averaging in a multiclass setting will produce precision, recall and  that are all identical to accuracy. Also note that “weighted” averaging may produce an F-score that is not between precision and recall.".(https://scikit-learn.org/stable/modules/model_evaluation.html)

"In other words, precision is the proportion of the predicted items that are relevant, and recall is the proportion of the relevant items that are correctly predicted." (p.231, mlandsecurity)

"A measurement can be accurate yet not precise, not accurate but still precise, neither accurate nor precise, or both accurate and precise. We consider a measurement to be valid if it is both accurate and precise." (p.39, deep learning)

""macro" simply calculates the mean of the binary metrics, giving equal weight to each class. In problems where infrequent classes are nonetheless important, macro-averaging may be a means of highlighting their performance. On the other hand, the assumption that all classes are equally important is often untrue, such that macro-averaging will over-emphasize the typically low performance on an infrequent class.
"weighted" accounts for class imbalance by computing the average of binary metrics in which each class’s score is weighted by its presence in the true data sample.
"micro" gives each sample-class pair an equal contribution to the overall metric (except as a result of sample-weight). Rather than summing the metric per class, this sums the dividends and divisors that make up the per-class metrics to calculate an overall quotient. Micro-averaging may be preferred in multilabel settings, including multiclass classification where a majority class is to be ignored." (https://scikit-learn.org/stable/modules/model_evaluation.html#average)

"The 1’s in each row denote the positive classes a sample has been labelled with." (https://scikit-learn.org/stable/modules/multiclass.html)

"F1 score of the positive class in binary classification or weighted average of the F1 scores of each class for the multiclass task." (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

## How to run:<a name="How_to_run"></a>
Assuming that your current working directory is home (i.e., data, webapp, models are all subdirectories one level below), you can then run the .py files "process_data.py", "train_classifier.py", & "run.py" in your terminal with the following commands. For example:
  
  ```bash
  python process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
  ```
  
  ```bash
  python train_classifier.py data/DisasterResponse.db models/classifier.pkl
  ```
  
  ```bash
  python webapp/run.py
  ```

Note: You might need to adjust the paths to run these .py files.

## Summary of results:<a name="Summary_of_results"></a>
work in progress

## Acknowledgements & licensing:<a name="Acknowledgements"></a>
- Thanks to FigureEight for their data on Twitter messages.
