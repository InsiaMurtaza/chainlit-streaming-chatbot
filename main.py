import os
from agents import Agent, Runner,OpenAIChatCompletionsModel
from openai import AsyncOpenAI
import chainlit as cl
from typing import cast
from dotenv import load_dotenv
from agents.run import RunConfig


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.0-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

@cl.on_chat_start
async def start():

    client = AsyncOpenAI(api_key=gemini_api_key, base_url= BASE_URL)

    config = RunConfig(model=OpenAIChatCompletionsModel(model=MODEL,openai_client= client),
                       model_provider= client,
                       tracing_disabled=True)
    cl.user_session.set("config", config)
    cl.user_session.set("chathistory",[])
    agent:Agent = Agent(name="Assistant", 
                        instructions="You're a helpful assistant", 
                        model= OpenAIChatCompletionsModel(model=MODEL,openai_client= client))
    cl.user_session.set("agent", agent)

    await cl.Message("Welcome to the Panaversity AI Assistant👋 How can I help you?").send()

@cl.on_message
async def main(message:cl.Message):

    msg = cl.Message(content="")
    await msg.send()

    agent:Agent = cast(Agent, cl.user_session.get("agent"))
    config = cast(RunConfig, cl.user_session.get("config"))

    history = cl.user_session.get("chathistory") or []
    history.append({"role":"user", "content": message.content})
    
    try:
        
        print("\n[CALLING AGENT WITH CONTEXT]\n",history,"\n")
        result = Runner.run_streamed(starting_agent= agent, input= history, run_config= config)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and hasattr(event.data,'delta'):
                token = event.data.delta
                await msg.stream_token(token)
                
        history.append({"role": "assistant", "content":msg.content})
        cl.user_session.set("chathistory", history)
        
        print(f"User: {message.content}")
        print(f"AI Assistant: {msg.content}")
        
    except Exception as e:
        await msg.update(content=f"Error: {str(e)}")
        print(f"Error: {str(e)}")
        

    
    

