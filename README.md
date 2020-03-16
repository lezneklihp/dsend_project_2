# Overview dsend_project_2:
- [Project summary](#Summary)
- [Repository content](#Repository_content)
- [Software requirements](#Software_requirements)
- [How to run locally](#How_to_run)
- [How to run locally in a Docker container](#How_to_run_with_docker)
- [Acknowledgements & licensing](#Acknowledgements)

## Project summary:<a name="Summary"></a>
Twitter allows its users to share concerns with the public. In the case of a disaster (such as the COVID-19 pandemic), making others aware of a problem can potentially save lives. When time is short, however, it could turn out to be challenging to interpret Twitter messages quickly.

The idea behind this project is thus to analyze text data & to categorize Twitter messages. Thereby rescuers can understand faster how to help people in need. This idea has been suggested originally by Figure Eight Inc.

From a technical perspective, this is a multilabel classification task of a supervised machine learning problem. In such tasks, samples can fall into more than one category. Evaluating whether a model does this correctly then turns out to be more complex than relying on the accuracy metric only. High accuracy scores might mislead us here. Thus, we also need to take precision & recall into account. At this point, a differentiation between accuracy & precision will help us evaluate the quality of a model's classifications.

In a few words, here is what this project entails:

1) Preparing data:
- read in .csv data
- create separate category columns
- remove characters & messages in their original language
- replace unexpected values for the "related" column
- drop any entries with NaNs
- remove duplicated entries
- cast binary data for categories to integer dtype
- save data to SQLite database

2) Training classifiers:
- load prepared data set
- preprocess text data with normalization, tokenization, stop word removal, stemming & lemmatization
- create machine learning pipeline with TF-IDF (i.e., proportional frequency of a word in a message), a classifier (here I used AdaBoost & LightGBM) & the MultiOutputClassifier (for multilabel classification)
- with this pipeline, grid search for the best parameter setting for each classifier
- evaluate classification results for each label with the F1-score
- save classifier to .pkl file

3) Making project accessible:
- adjust Flask (web) app to be able to define use of particular .pkl classifier file
- load classifier from .pkl file
- deploy Flask app locally & open with a browser
- type in messages in the user interface & view categorization result
- create Dockerfile to save others from having to adjust their Python development environment

And here is how the Flask app looks like.

Main page (top):
![file1](https://github.com/lezneklihp/dsend_project_2/blob/master/app/screenshots/main_page_top.png)

Main page (bottom):
![file2](https://github.com/lezneklihp/dsend_project_2/blob/master/app/screenshots/main_page_bottom.png)

Search results page:
![file3](https://github.com/lezneklihp/dsend_project_2/blob/master/app/screenshots/search_result_page.png)

## Repository content:<a name="Repository_content"></a>
This repository includes .csv, .db, .html, .pkl, .png & .py files for the second project of the Udacity Data Scientist for Enterprise Nanodegree. In addition, there are .sh, .txt & a Dockerfile which were created to make this project more easily transferable to other machines.

In "/data" there are all files needed for preparing data. In "/models" there are files for training classifiers. In "/app" there are files for running the Flask application. The other files in the root directory of this repository are used for creating a containerized version of the Flask app.

```bash
.
├── README.md
├── app
│   ├── run.py
│   ├── screenshots
│   │   ├── main_page_bottom.png
│   │   ├── main_page_top.png
│   │   └── search_result_page.png
│   └── templates
│       ├── go.html
│       └── master.html
├── data
│   ├── DisasterResponse.db
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   └── process_data.py
├── dockerfile
├── models
│   ├── ada_classifier.pkl
│   ├── ada_classifier_training_results.txt
│   ├── lgbm_classifier.pkl
│   ├── lgbm_classifier_training_result.txt
│   └── train_classifier.py
├── requirements.txt
└── setup.sh
```

## Software requirements:<a name="Software_requirements"></a>
Please use Python version 3.x & the following packages:

```bash
Flask==1.0
lightgbm==2.3.1
nltk==3.4.5
numpy==1.16.6
pandas==1.0.1
plotly==4.5.3
scikit-image==0.13.1
scikit-learn==0.19.1
scipy==0.19.1
SQLAlchemy==1.3.13
```

I used [pip](https://pip.pypa.io/en/stable/) to install these packages.

## How to run locally:<a name="How_to_run"></a>
Clone this repository to a directory. Then change directory into the git repository. Let's call the directory of the git repository "home".

Now assuming that your current working directory is home (i.e., app, data, models are all subdirectories one level below), you can then run the .py files "process_data.py", "train_classifier.py" & "run.py" in your terminal with the following commands. For example:

```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

```bash
python models/train_classifier.py data/DisasterResponse.db models/ada_classifier.pkl
```

```bash
python app/run.py ada_classifier.pkl
```

You can afterwards access the Flask app on http://0.0.0.0:3001/

Note: You might need to adjust the paths to run these .py files. The third command accepts file names for other .pkl files. In this repository, there are currently the following trained classifiers available:
- "ada_classifier.pkl" for the AdaBoostClassifier
- "lgbm_classifier.pkl" for the LGBMClassifier

## How to run locally in a Docker container:<a name="How_to_run_with_docker"></a>
If you want to run this app locally without (un-)installing current versions of Python packages on your machine, you can run the app inside a Docker container. Nonetheless, you will have to install Docker for this option in the first place.

Then clone this repository to a directory on your machine. Change directory to the git repository. There execute the following Docker commands to create a Docker image called "dsend-2" (or any other image name) in your terminal. If you don't want to access the Flask app on port 5001, specify a port to access the web app in a browser (e.g., Chrome). You will get your specific URL displayed in the terminal.

```bash
docker build -t dsend-2 .
```

```bash
docker run -p 3001:3001 -it --rm --name disasterresponse-app dsend-2:latest
```

Note: You can again load any other trained model as .pkl file with the entrypoint "setup.sh". For users with a Windows OS this Docker workaround might turn out to be difficult to put into practice.

## Acknowledgements & licensing:<a name="Acknowledgements"></a>
Thanks to Figure Eight for their data on Twitter messages.

Feel free to use this project, make changes and share your results!
