import azure.functions as func
import json, os, base64, logging
from shared.agentic_health import generate_meal_plan
from openai import OpenAI

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


# --- Text-based meal plan endpoint ---
@app.function_name(name="generate_meal_plan")
@app.route(route="generate_meal_plan", methods=["POST"])
def generate_meal_plan_fn(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
        goal = body.get("goal", "anti-inflammatory meal plan for prediabetes")
        result = generate_meal_plan(goal)
        return func.HttpResponse(json.dumps(result), mimetype="application/json", status_code=200)
    except Exception as e:
        logging.error(str(e))
        return func.HttpResponse(json.dumps({"error": str(e)}), status_code=500)


# --- Image upload + analysis endpoint ---
@app.function_name(name="analyze_photo")
@app.route(route="analyze_photo", methods=["POST"])
def analyze_photo(req: func.HttpRequest) -> func.HttpResponse:
    try:
        file = req.files.get("file")
        if not file:
            return func.HttpResponse("No file uploaded", status_code=400)

        img_bytes = base64.b64encode(file.stream.read()).decode()
        prompt = "Analyze this meal photo for glycemic impact and inflammation potential."

        # Try Azure OpenAI first
        if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            client = OpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
                default_headers={"api-version": "2024-05-01-preview"}
            )
        else:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a nutrition image analyzer."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_bytes}"}
                ]}
            ]
        )

        result = {"analysis": response.choices[0].message.content}
        return func.HttpResponse(json.dumps(result), mimetype="application/json", status_code=200)

    except Exception as e:
        logging.error(f"Image analysis failed: {e}")
        # Graceful fallback response
        return func.HttpResponse(
            json.dumps({
                "error": "Analysis failed. Check API quota or try offline mode.",
                "details": str(e)
            }),
            status_code=500,
            mimetype="application/json"
        )
