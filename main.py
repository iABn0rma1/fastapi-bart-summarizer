from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from transformers import pipeline
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.requests import Request

# Initialize the summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize FastAPI app
app = FastAPI()

# Set up template directory
templates = Jinja2Templates(directory="templates")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Route to render the HTML form for user input
@app.get("/", response_class=HTMLResponse)
async def get_upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# Summarize endpoint for both text and file uploads
@app.post("/summarize")
async def summarize_article(
    file: UploadFile = File(None),  # File upload is optional
    text: str = Form(None)          # Text input is optional
):
    # Handling file upload or text input
    if file:
        # If a file is uploaded, read and decode its content
        try:
            content = await file.read()
            article_text = content.decode('utf-8')
        except Exception:
            raise HTTPException(status_code=400, detail="File content must be readable as text")
    elif text:
        # If text input is provided
        article_text = text
    else:
        raise HTTPException(status_code=400, detail="Either upload a file or provide text content")

    # Ensure the article content is not empty
    if not article_text.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    # Generate summary using the BART model
    summary = summarizer(article_text, max_length=150, min_length=30, do_sample=False)

    # Return JSON response for API requests
    return JSONResponse(content={"summary": summary[0]['summary_text']})

