# AI Resume Polish

AI-powered resume correction and cover letter generator built with Flask.

## Features

- Resume text correction with Hugging Face API
- Grammar and spelling correction
- Generate personalized cover letters (Anschreiben) based on resume and job description
- Export results in PDF or TXT format
- Support for various file formats (PDF, DOCX, TXT)

## Installation

1. Clone this repository
2. Install dependencies using pip
3. Set up environment variables:
   - HUGGINGFACE_API_KEY - Your Hugging Face API token

## Usage

Run the application with:

```
gunicorn --bind 0.0.0.0:5000 main:app
```

Visit http://localhost:5000 in your browser.

## Deployment

This application can be deployed on platforms like Render, Heroku, or similar PaaS providers.

## License

MIT
