#!/bin/bash
python app/run.py ada_2_classifier.pkl
flask run --host 0.0.0.0
