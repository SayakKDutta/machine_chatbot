# Machine Chatbot
### Overview
#### Machine Chatbot is an advanced conversational AI application that integrates Pandas, ChatGroq's Llama3, and Chainlit frameworks to interact with CSV data. The chatbot can perform various data operations, such as querying, visualization, and analysis, all through natural language commands.
#### This chatbot is designed for businesses or analysts who need to extract insights from data in real-time without needing specialized coding knowledge. By allowing users to interact with datasets conversationally, Machine Chatbot simplifies the process of gaining actionable insights.
## Features used
#### Conversational AI: Interact with CSV data through natural language queries.
#### Data Visualization: Automatically generate visual plots, charts, or graphs based on user requests.
#### Custom Database Queries: Execute complex database queries based on user commands.
#### Tool-based Execution: The chatbot can dynamically call external tools like Pandas and Matplotlib to process and visualize the data.
## Architechture(the core files description and what they do)
### app.py:
#### Initializes and deploys the chatbot using Streamlit.
#### Manages user inputs, calls to the language model, and response generation.
#### Includes file uploader functionality for CSVs and handles the chat interface.
### bot.py:
#### Core logic for interacting with ChatGroq LLM.
#### Routes user queries to the appropriate tools, processes responses, and handles retries/errors.
#### Supports querying from databases and visualization.
### tools.py:
#### Contains helper functions to process data and generate charts/graphs.
#### Includes tools for querying SQL databases and manipulating data using Pandas.
### requirements.txt:
#### chainlit==1.2.0, langchain-groq, httpx==0.23.0, pyngrok, these are the libraries used here
## Technologies used
#### Langchain: Powers the logic for chaining together the different tools required for database querying and visualization.
#### ChatGroq: Integrates Llama3's large language model for processing natural language queries.
#### PandasAI: An AI-powered data manipulation library to interact with the data via Pandas DataFrames.
#### Streamlit: A framework for deploying the chatbot in a web-based interface.
#### Includes tools for querying SQL databases and manipulating data using Pandas.
## Flow of the project
### Prerequisites
#### Installed python 3.8+
#### Pip installed python file
#### Reposetory has been cloned
#### Installed dependencies
#### Set up Environment Variables through chatgroq API key
## Usage
#### Uploading CSV Files
#### Example Queries
#### Visualizations
## Deployment
#### To run the app on chainlit [DEPLOY](DEPLOY_with_Collab(ngrok).ipynb) 
## Max sql query shown on screen inchainlit is 10
## Charts included are line, scatter and bar chart
## Sample questions: 
#### Give me the first 5rows of the data
#### Give a table of product id and air temperature with last five rows of the table
#### Give a bar chart of product id and air temperature with last five rows of the table
#### Average process temperature
#### What is OSF
## Data sources are there in the [src](https://github.com/milichowdhury0096/Machine_Chatbot/tree/main/src) file
## License
#### This project is licensed under the MIT License. You can find the details in the [LICENSE](https://github.com/milichowdhury/Machine_Chatbot/blob/main/LICENSE) file.
## Acknowledgments
#### Langchain for tool-based architecture.
#### ChatGroq for the powerful Llama3 language model.
