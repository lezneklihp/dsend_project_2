#!/bin/bash
python train_classifier.py data/DisasterResponse.db models/ada_classifier.pkl
python webapp/run.py ada_classifier.pkl
flask run --host 0.0.0.0
