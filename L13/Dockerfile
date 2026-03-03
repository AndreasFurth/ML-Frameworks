# Exempeldockerfile för en FastAPI-app som servar en ML-modell

# Först sätter vi vad vi vill köra vår kod i för en miljö
# En pythonversion är rimligt
FROM python:3.11-slim 

# Vi sätter vart på datorn vi vill jobba
WORKDIR /app

# Vi kopierar materialet där vi är till arbetsytan på den 
# virtuella datorn
COPY . /app

# Vi installerar programmen vi behöver 
# OFTA: från requirements.txt
# rimligtvis går från uv?
RUN pip install --no-cache-dir fastapi uvicorn joblib scikit-learn numpy

# Exponerar en port
EXPOSE 8000

# Vad som faktiskt ska göras när containern kör
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]