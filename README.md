MLOps project - Car accident
==============================

This project is realized during the formation "MLOps" at Datascientest. 
It aims to develop and test a machine learning model and create an API to access the model and make real time predictions.
The raw data is loaded in data/raw (added to gitignore because of the total size) before being processed.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original data dump, taken from data.gouv.fr
    │
    ├── logs               <- Logs from the API tests (added to gitignore)
    │
    ├── models             <- Trained models and encoders used int eh API to make predictions
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │                         Used to explore data and test models before creating the API
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── api            <- Scripts create the API and the API Container
    │   │   └── app.py
    │   │   └── auth.py
    │   │   └── log_module.py
    │   │   └── Dockerfile_api
    │   │   └── api-deployment.yaml
    │   │   └── service-api.yaml
    │   │
    │   ├── dags            <- Airflow module
    │   │   └── pipeline_dag.py.py
    │   │
    │   ├── data           <- Scripts to download or generate data and create the Data Container
    │   │   ├── make_dataset.py
    │   │   ├── build_features.py
    │   │   ├── config.py
    │   │   ├── etl.py
    │   │   └── Dockerfile_data
    │   │
    │   ├── k8s            <- Kubernetes files
    │   │   ├── api-deployment.yaml
    │   │   ├── data-deployment.yaml
    │   │   ├── models-deployment.yaml
    │   │   ├── persistent-volume-claim.yaml
    │   │   ├── persistent-volume.yaml
    │   │   ├── service-api.yaml
    │   │   ├── service-data.yaml
    │   │   └── service-models.yaml
    │   │
    │   ├── models         <- Scripts to train models and create the Model Container
    │   │   ├── predict_model.py
    │   │   ├── train_model.py
    │   │   ├── config.py
    │   │   ├── model_pipeline.py
    │   │   └── Dockerfile

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
