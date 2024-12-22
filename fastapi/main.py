from fastapi import FastAPI, HTTPException
import subprocess
import json
from typing import Dict, Any

app = FastAPI()


@app.post("/chat")
async def predict(request: Dict[str, Any]):
    prompt = request.get("message", "")  # Get the prompt (message) from the request body
    try:
        # Run the Llama model using Ollama CLI
        process = subprocess.Popen(
            ["ollama", "run", "llama3.2:1b"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )

        stdout, stderr = process.communicate(input=prompt)
        
        # Log the model's stdout and stderr for debugging
        print("stdout:", stdout)
        print("stderr:", stderr)

        if process.returncode != 0:
            raise HTTPException(status_code=500, detail={"error": "Failed to run Llama model", "details": stderr})
        
        # Assuming the model output is in plain text
        response = stdout.strip()
        return {"response": response}

    except HTTPException as http_ex: # Catch HTTP Exception if raised
         raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Something went wrong", "details": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11434)