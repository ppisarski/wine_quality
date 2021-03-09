# Wine Quality Modeling

## Introduction 
TODO: Give a short introduction of your project. Let this section explain the objectives or the motivation behind this project. 

## Reading

- [A better way to build ML — why you should be using Active Learning](https://humanloop.com/blog/why-you-should-be-using-active-learning/)
- [Build A Web App To Interpret Your ML Predictions in Seconds With Shapash](https://pub.towardsai.net/build-a-web-app-to-interpret-your-ml-predictions-in-seconds-with-shapash-e2ddb2df4d02)

## EDA

## Explainable AI

Explainable AI makes machine learning interpretable, so that models can be understood by humans and assist in decision making.
There are number of tools available:
SHAP and LIME are two most popular packages, others include
Skater, Interpret ML, etc.





## Training the model

1. Run `python -m src.data_cleanup` to create dataset which should be saved in `input/data.csv`.
2. Run `python -m src.create_folds` 
3. Run `./train.sh` to train the model (remember to tune the model).
All models will be saved in `models` directory.

# Streamlit Application

To run streamlit application

    streamlit run app/app.py 

To build docker image

    docker build --tag completions .
    
To run docker image as a container on local machine

    docker run --publish 8080:80 --detach completions
    
The app will be available at http://localhost:8080

To build docker image and save it to Azure Container Registry

    az acr build --registry sprouleregistry --resource-group SprouleAppsRG --image completions-app .
 
The app will be available at https://<webapp_name>.azurewebsites.net

# Reading

- [Deploy a Streamlit Web App with Azure App Service](https://towardsdatascience.com/deploying-a-streamlit-web-app-with-azure-app-service-1f09a2159743)


- https://stackoverflow.com/questions/18733965/annoying-message-when-opening-windows-from-python-on-os-x-10-8