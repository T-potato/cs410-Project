from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from generate_summary import generate_summary
import uvicorn
import logging
# Initialize FastAPI app
app = FastAPI()

logger = logging.getLogger("fastapi")
logger.setLevel(logging.DEBUG)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific domains if needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body model
class SummaryRequest(BaseModel):
    text: str
    model_type: str = "bart"  # Optional: Default to "bart"
    max_summary_length: int = 150  # Optional: Default to 150

# Inference route
@app.post("/")
async def inference(request: SummaryRequest):
    try:
        # Generate summary
        summary = generate_summary(
            article=request.text,
            model_type=request.model_type,
            max_input_length=1024,
            max_summary_length=request.max_summary_length,
        )
        return {"summary": summary}
    except Exception as e:
        # Handle errors and return an HTTP exception
        logger.error(f"Error in inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level='debug')