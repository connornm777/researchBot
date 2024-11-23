from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import os
import gpt_tools as gt
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))




