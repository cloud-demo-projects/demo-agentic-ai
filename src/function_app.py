import azure.functions as func
import json, os, base64, logging
from shared.agentic_health import generate_meal_plan
import base64
import json
import logging
import os
from threading import Lock
from typing import Optional, Tuple
from openai import OpenAI
from shared.agentic_health import generate_meal_plan

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


MAX_JSON_BYTES = int(os.getenv("MAX_JSON_BYTES", 16 * 1024))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", 4 * 1024 * 1024))

_IMAGE_CLIENT_LOCK = Lock()
_IMAGE_CLIENT: Optional[OpenAI] = None
_IMAGE_CLIENT_SIGNATURE: Optional[Tuple[str, str, str, str, str]] = None
_IMAGE_MODEL: Optional[str] = None


def _current_image_signature() -> Tuple[str, str, str, str, str]:
    """Build a signature that reflects the active credentials/configuration."""

    return (
        os.getenv("AZURE_OPENAI_API_KEY", ""),
        os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        os.getenv("OPENAI_API_KEY", ""),
        os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )


def _resolve_image_client() -> Tuple[OpenAI, str]:
    """Return a cached OpenAI client plus the model to use for vision calls."""

    global _IMAGE_CLIENT, _IMAGE_MODEL, _IMAGE_CLIENT_SIGNATURE

    signature = _current_image_signature()

    with _IMAGE_CLIENT_LOCK:
        if _IMAGE_CLIENT and _IMAGE_CLIENT_SIGNATURE == signature and _IMAGE_MODEL:
            return _IMAGE_CLIENT, _IMAGE_MODEL

        azure_key, azure_endpoint, azure_deployment, openai_key, openai_model = signature

        if azure_key and azure_endpoint:
            logging.info("Using Azure OpenAI vision deployment %s", azure_deployment)
            client = OpenAI(
                api_key=azure_key,
                base_url=azure_endpoint,
                default_headers={"api-version": "2024-05-01-preview"},
            )
            model = azure_deployment
        elif openai_key:
            logging.info("Using OpenAI vision model %s", openai_model)
            client = OpenAI(api_key=openai_key)
            model = openai_model or "gpt-4o-mini"
        else:
            raise RuntimeError("No OpenAI credentials configured for image analysis")

        _IMAGE_CLIENT = client
        _IMAGE_MODEL = model
        _IMAGE_CLIENT_SIGNATURE = signature

        return client, model


def _get_cors_headers():
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
    return {
        "Access-Control-Allow-Origin": allowed_origins,
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type,Authorization",
        "Access-Control-Max-Age": "86400",
    }


def _cors_response(body="", status_code=200, mimetype="application/json"):
    return func.HttpResponse(
        body=body,
        status_code=status_code, 
        mimetype=mimetype,
        headers=_get_cors_headers()
    )


# --- Text-based meal plan endpoint ---
@app.function_name(name="generate_meal_plan")
@app.route(route="generate_meal_plan", methods=["POST", "OPTIONS"])
def generate_meal_plan_fn(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return _cors_response(status_code=204)

    try:
        body = req.get_json()
        raw_body = req.get_body() or b"{}"
        if len(raw_body) > MAX_JSON_BYTES:
            logging.warning("Rejected request body over %s bytes", MAX_JSON_BYTES)
            return _cors_response(
                json.dumps({"error": "Request body too large"}),
                status_code=413
            )

        try:
            body = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON payload"}),
                status_code=400,
                mimetype="application/json",
            )

        goal = body.get("goal")
        purpose = body.get("purpose")
        result = generate_meal_plan(goal, purpose)
        return _cors_response(json.dumps(result))
    except Exception as e:
        logging.error(str(e))
        return _cors_response(json.dumps({"error": str(e)}), status_code=500)


# --- Image upload + analysis endpoint ---
@app.function_name(name="analyze_photo") 
@app.route(route="analyze_photo", methods=["POST", "OPTIONS"])
def analyze_photo(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return _cors_response(status_code=204)

    try:
        file = req.files.get("file")
        if not file:
            return _cors_response(
                json.dumps({"error": "No file uploaded"}), 
                status_code=400
            )

        img_bytes = base64.b64encode(file.stream.read()).decode()
        declared_size = getattr(file, "content_length", None)
        if declared_size and declared_size > MAX_IMAGE_BYTES:
            logging.warning("Rejected upload over %s bytes (declared)", MAX_IMAGE_BYTES)
            return func.HttpResponse("File too large", status_code=413)

        image_bytes = file.stream.read(MAX_IMAGE_BYTES + 1)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            logging.warning("Rejected upload over %s bytes (actual)", MAX_IMAGE_BYTES)
            return func.HttpResponse("File too large", status_code=413)

        mimetype = getattr(file, "mimetype", "") or ""
        if mimetype and not mimetype.startswith("image/"):
            return func.HttpResponse("Unsupported file type", status_code=415)

        img_bytes = base64.b64encode(image_bytes).decode()
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
        client, model = _resolve_image_client()

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
        return _cors_response(json.dumps(result))
    except Exception as e:
        logging.error(f"Image analysis failed: {e}")
        return _cors_response(
            json.dumps({
                "error": "Analysis failed. Check API quota or try offline mode.",
                "details": str(e)
            }),
            status_code=500
        )