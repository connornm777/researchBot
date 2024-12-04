from openai import OpenAI
import numpy as np
from dotenv import dotenv_values
import os
import utils as gt
env_vars = dotenv_values()

class test:
    def __init__(self):
        for key, value in dotenv_values().items():
            setattr(self, key, value)



