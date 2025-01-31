import asyncio
import logging
import os
import json

from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO)

# Define DeepSeek Model
model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
client = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0,
    seed=1,
    base_url="http://54.174.178.103:5010/v1"
)

# Main chatbot class
class ChatBot:
    def __init__(self, system, tools, tool_functions):
        self.system = system
        self.tools = tools
        self.exclude_functions = ["plot_chart"]
        self.tool_functions = tool_functions
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    async def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        response_message = await self.execute()
        
        if response_message["content"]:
            self.messages.append({"role": "assistant", "content": response_message["content"]})

        logging.info(f"User message: {message}")
        logging.info(f"Assistant response: {response_message['content']}")

        return response_message

    async def execute(self):
        response = await deepseek_model.agenerate(messages=self.messages)
        assistant_message = response.generations[0][0].text  # Extracting generated text
        
        return {"role": "assistant", "content": assistant_message}

    async def call_function(self, tool_call):
        function_name = tool_call["function"]["name"]
        function_to_call = self.tool_functions[function_name]
        function_args = json.loads(tool_call["function"]["arguments"])

        logging.info(f"Calling {function_name} with {function_args}")
        function_response = await function_to_call(**function_args)

        return {
            "role": "tool",
            "name": function_name,
            "content": function_response,
        }

    async def call_functions(self, tool_calls):
        function_responses = await asyncio.gather(
            *(self.call_function(tool_call) for tool_call in tool_calls)
        )

        responses_in_str = [{**item, "content": str(item["content"])} for item in function_responses]

        for res in function_responses:
            logging.info(f"Tool Call: {res}")

        self.messages.extend(responses_in_str)

        response_message = await self.execute()
        return response_message, function_responses
