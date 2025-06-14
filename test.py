import os
from dotenv import load_dotenv

if load_dotenv():
    print(os.environ.get("NAME"))
else:
    print("Failed")