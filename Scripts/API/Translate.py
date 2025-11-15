from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import logging
from deep_translator import GoogleTranslator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vietnamese Translation API",
    description="API for translating Vietnamese text to English using Google Translator",
    version="1.0.0",
    root_path="/siu_sayan_6"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class TranslateRequest(BaseModel):
    text: str
    source: Optional[str] = "vi"
    target: Optional[str] = "en"

class TranslateResponse(BaseModel):
    translated_text: str
    original_text: str
    source_language: str
    target_language: str

class HealthResponse(BaseModel):
    status: str
    service: str

@app.post("/translate", response_model=TranslateResponse)
async def translate_post(request: TranslateRequest):
    """Translate text using POST method"""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided or text is empty")
    
    try:
        logger.info(f"Translating text from {request.source} to {request.target}: {request.text[:50]}...")
        
        # Use Google Translator
        translator = GoogleTranslator(source=request.source, target=request.target)
        translated_text = translator.translate(request.text)
        
        if not translated_text:
            raise HTTPException(status_code=500, detail="Translation failed - empty result")
        
        return TranslateResponse(
            translated_text=translated_text,
            original_text=request.text,
            source_language=request.source,
            target_language=request.target
        )
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

@app.get("/translate", response_model=TranslateResponse)
async def translate_get(
    text: str = Query(..., description="Text to translate"),
    source: str = Query("vi", description="Source language code (default: vi)"),
    target: str = Query("en", description="Target language code (default: en)")
):
    """Translate text using GET method"""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No text provided or text is empty")
    
    try:
        logger.info(f"Translating text from {source} to {target}: {text[:50]}...")
        
        # Use Google Translator
        translator = GoogleTranslator(source=source, target=target)
        translated_text = translator.translate(text)
        
        if not translated_text:
            raise HTTPException(status_code=500, detail="Translation failed - empty result")
        
        return TranslateResponse(
            translated_text=translated_text,
            original_text=text,
            source_language=source,
            target_language=target
        )
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="Vietnamese Translation API"
    )

@app.get("/")
async def index():
    """Root endpoint with API information"""
    return {
        "message": "Vietnamese Translation API",
        "version": "1.0.0",
        "endpoints": {
            "POST /translate": "Translate text using POST method",
            "GET /translate": "Translate text using GET method",
            "GET /health": "Health check",
            "GET /": "This endpoint"
        },
        "default_translation": "Vietnamese (vi) → English (en)",
        "status": "ready"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Vietnamese Translation API server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8506,
        log_level="info"
    )