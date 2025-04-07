from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

# Define a Pydantic model for the input data
class DateTimeInput(BaseModel):
    date_time: datetime

class AnotherInput(BaseModel):
    input_value: str
    date: datetime

# Endpoint 1: User gives input for date and time and returns a number
@app.post("/fetch-previous-footfall-count/")
async def get_number_from_datetime(data: DateTimeInput):
    # Example logic: return the timestamp as a number
    return {"number": int(data.date_time.timestamp())}

# Endpoint 2: User gives another input and date and gets back a number
@app.post("/predict-footfall-count/")
async def get_number_from_input_and_date(data: AnotherInput):
    # Example logic: return the length of the input string plus the day of the month
    number = len(data.input_value) + data.date.day
    return {"number": number}


# Run the application
# Use `uvicorn filename:app --reload` to run the FastAPI app