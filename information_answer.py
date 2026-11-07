import requests
import json

def generate_information(prompt):
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "phi3",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        # Extended timeout to 120 seconds for non-English script generation
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code == 200:
            res_json = response.json()
            return res_json.get("response", "")
        else:
            return f"Error: Received status code {response.status_code}"
            
    except requests.exceptions.Timeout:
        return "Error: The AI model took too long to respond. Please try again."
    except Exception as e:
        return f"Error connecting to Ollama: {e}"