#!/bin/bash
python process_data.py data/disaster_messages.csv data/disaster_messages.csv data/DisasterResponse.db
python train_classifier.py data/DisasterResponse.db models/ada_classifier.pkl
python webapp/run.py ada_classifier.pkl
flask run --host 0.0.0.0
