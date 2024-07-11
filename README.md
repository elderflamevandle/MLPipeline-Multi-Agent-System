# MLPipeline Multi-Agent System

This repository contains a Python script for a machine learning pipeline that utilizes multi-agent collaboration to perform classification and regression tasks on datasets. The script leverages the PyCaret library for machine learning tasks and the Langchain framework for agent communication.


## Introduction

The `MLPipeline` class is designed to automate the process of performing classification and regression tasks on datasets using a multi-agent system. The system includes a user proxy agent, a product manager agent, and a machine learning engineer agent that work together to complete tasks based on user prompts.

## Features

- Automated classification and regression tasks using PyCaret.
- Multi-agent collaboration for task execution.
- Configurable through environment variables.
- Supports Azure OpenAI for natural language processing.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/mlpipeline-multiagent.git
   cd mlpipeline-multiagent

2. Create and activate a virtual environment:

    ```sh
    python3 -m venv venv
    source venv/bin/activate  
    # On Windows, use `venv\Scripts\activate`

3. Install the required packages:

    ```sh
    pip install -r requirements.txt

## Usage
Ensure you have set the necessary environment variables (see below).

Modify the main() function in the script to suit your dataset and requirements.

Run the script:
`python main.py`

## Environment Variables
The script uses the following environment variables:

- AZURE_API_KEY: Your Azure OpenAI API key.
- BASE_URL: The base URL for the Azure OpenAI API.
- API_TYPE: The type of API being used (default is azure).
- API_VERSION: The version of the API (default is 2023-03-15-preview).

##  Agents

### User Proxy Agent
The UserProxyAgent interacts with the user and initiates chats with other agents to solve problems.

### Product Manager Agent
The ProductManager agent suggests a plan, prioritizes solutions, and creates a roadmap for machine learning projects.

### Machine Learning Engineer Agent
The MLEngineer agent analyzes tasks, selects the appropriate machine learning tool (classification or regression), and executes the required functions.

## Functions
### Classification_task
Performs a classification task on a given dataset.

> Parameters:
- data_path: The path to the CSV file containing the dataset.
- unseen_data_path: The path where unseen data is stored.
- target_column: The name of the target column in the dataset.
- save_dir: The path where the trained model is saved and loaded from.
- regression_task
Performs a regression task on a given dataset.

