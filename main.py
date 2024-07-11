import os
import tempfile
import pickle
from dotenv import load_dotenv
import pandas as pd
from pycaret.classification import *
from pycaret.regression import *
from langchain_openai import AzureOpenAI
from langchain.prompts import PromptTemplate
from autogen import UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager
from autogen.coding.local_commandline_code_executor import LocalCommandLineCodeExecutor
# Set environment variables
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://madhukar-kumar.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_KEY"] = "bd18995c51fa40e19e493df21c7ded81"
os.environ["DEPLOYMENT_NAME"] = "UTCL"



class MLPipeline:
    def __init__(self):
        self.config_list = {
                "model": "gpt-35-turbo-16k",
                "api_key": "bd18995c51fa40e19e493df21c7ded81",
                "base_url": "https://madhukar-kumar.openai.azure.com/",
                "api_type": "azure",
                "api_version": "2023-03-15-preview"
            }

        self.llm_config = {"config_list": [self.config_list], "timeout": 120, "cache_seed": 100}

        self.llm_config_ml_engineer = {
            "functions": [
                {
                    "name": "classification_task",
                    "description": "Perform a classification task on a given dataset",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data_path": {
                                "type": "string",
                                "description": "The path to the CSV file containing the dataset"
                            },
                            "unseen_data_path": {
                                "type": "string",
                                "description": "The path where unseen data is stored."
                            },
                            "target_column": {
                                "type": "string",
                                "description": "The name of the target column in the dataset"
                            },
                            "save_dir": {
                                "type": "string",
                                "description": "The path where training model saved and loaded from"
                            }
                        },
                        "required": ["data_path", "unseen_data_path", "target_column", "save_dir"]
                    }
                },
                {
                    "name": "regression_task",
                    "description": "Perform a Regression task on a given dataset",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data_path": {
                                "type": "string",
                                "description": "The path to the CSV file containing the dataset"
                            },
                            "unseen_data_path": {
                                "type": "string",
                                "description": "The path where unseen data is stored."
                            },
                            "target_column": {
                                "type": "string",
                                "description": "The name of the target column in the dataset"
                            },
                            "save_dir": {
                                "type": "string",
                                "description": "The path where training model saved and loaded from"
                            }
                        },
                        "required": ["data_path", "unseen_data_path", "target_column", "save_dir"]
                    }
                },
            ],
            "config_list": [self.config_list],
        }

    @staticmethod
    def classification_task(data_path,unseen_data_path, target_column, save_dir):
        data = pd.read_csv(data_path)
        df_unseen = pd.read_csv(unseen_data_path)
        expt = ClassificationExperiment()
        expt.setup(data, target=target_column, session_id=123)
        print('@@@@@@@@@@@@@@@@@@@@@@@@')
        best_model = expt.create_model('rf')
        print('!!!!!!!!!!')
        tuned_model = expt.finalize_model(best_model)
        print('************')
        saved_model = expt.save_model(tuned_model, os.path.join(save_dir, 'Saved_class_model'))
        model = expt.load_model(os.path.join(save_dir, 'Saved_class_model'))
        predictions = expt.predict_model(model, data=df_unseen)
        predictions = predictions.sort_values("prediction_score", ascending= False).reset_index()
        return predictions.head(10), expt

    @staticmethod
    def regression_task(data_path, unseen_data_path, target_column, save_dir):
        data = pd.read_csv(data_path)
        df_unseen = pd.read_csv(unseen_data_path)
        expt = RegressionExperiment()
        expt.setup(data, target=target_column, session_id=123)
        best_model = expt.create_model('lr')
        tuned_model = expt.finalize_model(best_model)
        saved_model = expt.save_model(tuned_model, os.path.join(save_dir, 'Saved_class_model'))
        model = expt.load_model(os.path.join(save_dir, 'Saved_class_model'))
        predictions = expt.predict_model(model, data=df_unseen)
        predictions = predictions.sort_values("prediction_score", ascending= False).reset_index()
        return predictions.head(10), expt

    @staticmethod
    def predictions(save_dir, unseen_data_path, expt):
        if unseen_data_path:
            df = pd.read_csv(unseen_data_path)
            model = expt.load_model(os.path.join(save_dir, 'Saved_class_model2'))
            predictions = expt.predict_model(model, data=df)
            return predictions
        else:
            return "No Unseen Data Provided"

    def user_response(self, user_message, dataset_path, saving_dir, target_column, unseen_data_path):
        chat_res = self.user_proxy.initiate_chat(
            self.manager,
            message=f"""
                Consider the User Prompt mentioned in the triple quotes and based upon the same, work with agents to solve the problem.
                User Prompt : ```{user_message}```.
                Consider following column as a target variable.
                Target Variable : ```{target_column}```
                For this usecase consider following dataset path:
                DATASET PATH: ```{dataset_path}```
                TO save model and model results use following path:
                SAVING PATH: ```{saving_dir}```
                Consider a fllowing path for Unseen Dataset:
                UNSEEN DATA PATH: ```{unseen_data_path}```
            """,
        )
        return chat_res

    def setup_agents(self):
        self.user_proxy = UserProxyAgent(
            name="Admin",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            system_message="A human admin. Interact with the Product Manager to discuss the plan. Plan execution needs to be approved by this admin.",
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config={"use_docker": False},
            function_map={
                "classification_task": self.classification_task,
                "regression_task": self.regression_task,
                },
        )

        self.product_manager = AssistantAgent(
            name="Product_Manager",
            system_message="""Product_Manager. Your role is to suggest a plan, prioritize solutions, and create a roadmap for a machine learning project.
                      You will work with one agents: ML_Engineer. Here are their responsibilities:

                        ML_Engineer:
                        - Analyze the given task or problem statement to determine if it's a regression or classification task.
                        - Select the appropriate tool (regression or classification) from the available set of skills. Only choose among them, do not choose any other tool randomly.
                        - Retrieve the JSON object for the selected tool, parse the required parameters, and execute the code mentioned in the tool's function with the required parameters, using the provided executor.

                        Your job as the Product_Manager is to:
                        1. Explain the overall plan, clearly specifying which steps will be performed by the ML_Engineer and Tester.
                        2. After Tester Work is Completed you task is to give output and Output must contain the explanation of predictions either given by Tester ot ML_Engineer.

                        Ensure that your plan covers the end-to-end process, from ML_Engineer.

                        Explain the Dataframe Received from ML engineer which is an output from given set of tools. Explain that dataframe along with predictions and prediction probability.

                        """,
            llm_config=self.llm_config,
        )

        self.ml_engineer = AssistantAgent(
            name="ML_Engineer",
            llm_config=self.llm_config_ml_engineer,
            system_message="""ML_Engineer. You have access to two tools for machine learning tasks: a regression tool and a classification tool. Each tool is defined as a JSON object with its function, description, and required parameters. Your job is to:

                        1. Analyze the given task or problem statement.
                        2. Determine whether the problem is a regression task or a classification task.
                        3. Based on the problem type, select the appropriate tool (regression or classification) from the available set of tools.
                        4. Retrieve the JSON object for the selected tool and parse the required parameters for its function.
                        5. Execute the code mentioned in the tool's function with the required parameters, using the provided executor. Do not write any code from your side, only use provided code.
                        6. Check the execution result returned by the executor. If there is an error, try to fix it by adjusting the parameters or revisiting your analysis of the problem type.
                        7. Reply with TERMINATE when task is completed.

                        You must follow the approved plan and ensure that you use correct column names after encoding, if applicable. Do not write any code yourself; only execute the code provided in the selected tool. Do not use a code block if it's not intended
                        to be executed by the executor. Do not include multiple code blocks in one response. Do not ask others to copy and paste the result. Do not use all the functions , only use one function which is intended for the task.""",
        )

        self.groupchat = GroupChat(agents=[self.user_proxy, self.product_manager, self.ml_engineer], messages=[],
                                   max_round=10)
        self.manager = GroupChatManager(groupchat=self.groupchat, llm_config=self.llm_config)


def main():
    user_message = f"""Give me 10 employees with high attrition probability"""
    dataset_path = "employee.csv"
    saving_dir = "coding01"
    target_column = "Attrition"
    unseen_data_path = ""

    if not unseen_data_path:
        unseen_data_path = dataset_path

    ml_pipeline = MLPipeline()
    ml_pipeline.setup_agents()

    print("dataset_path -->", dataset_path)
    print("unseen_dath_path -->", unseen_data_path)


    response = ml_pipeline.user_response(user_message, dataset_path, saving_dir, target_column, unseen_data_path)

    print(response)


if __name__ == "__main__":
    main()