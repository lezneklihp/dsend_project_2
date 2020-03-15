#!/bin/bash
python app/run.py lgbm_classifier.pkl
flask run --host 0.0.0.0
