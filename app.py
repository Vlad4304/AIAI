import os
import io
import logging
import traceback
import sys
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from werkzeug.utils import secure_filename
import tempfile
import json
from resume_parser import parse_resume_file, parse_resume_text
from ai_analyzer import analyze_resume, generate_anschreiben, score_resume

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Store session data in files instead of cookies to handle larger data
from flask_session import Session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(tempfile.gettempdir(), 'flask_sessions')
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
Session(app)

# Configure upload settings
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
TEMP_FOLDER = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Clear any previous CV data from session
    if 'resume_text' in session:
        session.pop('resume_text')
    if 'corrections' in session:
        session.pop('corrections')
    if 'anschreiben' in session:
        session.pop('anschreiben')
    if 'job_description' in session:
        session.pop('job_description')
    if 'resume_score' in session:
        session.pop('resume_score')
    
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    resume_text = ""
    
    # Check if the user uploaded a file or pasted text
    if 'resume_file' in request.files and request.files['resume_file'].filename:
        file = request.files['resume_file']
        
        logging.debug(f"File upload detected: {file.filename}")
        
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload a PDF, DOCX, or TXT file.', 'danger')
            return redirect(url_for('index'))
        
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(TEMP_FOLDER, filename)
            logging.debug(f"Saving file to: {file_path}")
            file.save(file_path)
            
            # Parse the CV file
            logging.debug(f"Parsing file: {file_path}")
            resume_text = parse_resume_file(file_path)
            logging.debug(f"File parsed successfully, text length: {len(resume_text)}")
            
            # Remove the temporary file
            os.remove(file_path)
            
        except Exception as e:
            flash(f'Error parsing file: {str(e)}', 'danger')
            logging.error(f"File parsing error: {str(e)}")
            return redirect(url_for('index'))
    
    elif request.form.get('resume_text'):
        resume_text = request.form.get('resume_text')
        resume_text = parse_resume_text(resume_text)
    
    else:
        flash('Please upload a file or paste your CV text.', 'warning')
        return redirect(url_for('index'))
    
    # Check if CV text is too short
    if len(resume_text) < 50:
        flash('The CV content is too short or could not be properly extracted. Please try again.', 'warning')
        return redirect(url_for('index'))
    
    try:
        # Get language preference from form
        language = request.form.get('language', 'en')
        
        # Analyze the CV with AI using specified language
        corrections = analyze_resume(resume_text, language)
        
        # Store the CV, corrections, and language in the session
        # Limit the number of corrections to prevent session size issues
        max_corrections = 20
        if len(corrections) > max_corrections:
            logging.warning(f"Limiting corrections from {len(corrections)} to {max_corrections} to prevent session overflow")
            corrections = corrections[:max_corrections]
            
        session['resume_text'] = resume_text
        session['corrections'] = corrections
        session['language'] = language
        
        return redirect(url_for('results'))
        
    except Exception as e:
        flash(f'Error analyzing CV: {str(e)}', 'danger')
        logging.error(f"AI analysis error: {str(e)}")
        return redirect(url_for('index'))

@app.route('/results')
def results():
    # Check if CV and corrections are in session
    if 'resume_text' not in session or 'corrections' not in session:
        flash('Please submit a CV for analysis first.', 'warning')
        return redirect(url_for('index'))
    
    resume_text = session['resume_text']
    corrections = session['corrections']
    
    # Get CV score if not already in session
    score_data = session.get('resume_score', None)
    if not score_data:
        try:
            # Get the language preference from the session
            language = session.get('language', 'en')
            score_data = score_resume(resume_text, language)
            session['resume_score'] = score_data
        except Exception as e:
            logging.error(f"Error scoring CV: {str(e)}")
            score_data = {
                "overall": 0,
                "categories": {
                    "content": 0,
                    "format": 0,
                    "language": 0,
                    "conciseness": 0
                },
                "summary": "Error in evaluation" if language == 'en' else "Fehler bei der Bewertung"
            }
    
    return render_template('result.html', resume_text=resume_text, corrections=corrections, score=score_data)

@app.route('/download', methods=['POST'])
def download():
    if 'resume_text' not in session:
        flash('No CV data available for download.', 'warning')
        return redirect(url_for('index'))
    
    # Get the corrected CV text from the form
    corrected_text = request.form.get('corrected_text', '')
    
    if not corrected_text:
        flash('No content to download.', 'warning')
        return redirect(url_for('results'))
    
    # Create a temporary file with the corrected CV
    format_type = request.form.get('format', 'txt')
    
    if format_type == 'txt':
        # Create a text file
        file_buffer = io.BytesIO()
        file_buffer.write(corrected_text.encode('utf-8'))
        file_buffer.seek(0)
        
        return send_file(
            file_buffer,
            as_attachment=True,
            download_name='corrected_cv.txt',
            mimetype='text/plain'
        )
    else:  # PDF
        try:
            from fpdf import FPDF
            
            # Create a PDF file
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            # Clean text of non-compatible characters 
            # Replace special characters with standard ASCII alternatives
            clean_text = corrected_text.replace('\u2013', '-')  # en-dash
            clean_text = clean_text.replace('\u2014', '--')     # em-dash
            clean_text = clean_text.replace('\u2018', "'")      # left single quote
            clean_text = clean_text.replace('\u2019', "'")      # right single quote
            clean_text = clean_text.replace('\u201c', '"')      # left double quote
            clean_text = clean_text.replace('\u201d', '"')      # right double quote
            
            # Split the text into lines and add to PDF
            for line in clean_text.split('\n'):
                # Handle potential Unicode errors with a general filter
                safe_line = ''.join(c if ord(c) < 128 else '-' for c in line)
                pdf.multi_cell(0, 10, safe_line)
            
            # Save the PDF to a memory buffer
            file_buffer = io.BytesIO()
            
            try:
                # Get the PDF as bytes
                pdf_data = pdf.output(dest='S')
                if isinstance(pdf_data, str):
                    pdf_data = pdf_data.encode('latin-1')
                
                file_buffer.write(pdf_data)
                file_buffer.seek(0)
            except UnicodeEncodeError:
                logging.warning("Unicode encode error when creating PDF, using ASCII fallback")
                # If still encountering unicode errors, fall back to ASCII only
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                
                # Use ASCII-only version
                ascii_text = corrected_text.encode('ascii', 'replace').decode('ascii')
                for line in ascii_text.split('\n'):
                    pdf.multi_cell(0, 10, line)
                
                pdf_data = pdf.output(dest='S')
                if isinstance(pdf_data, str):
                    pdf_data = pdf_data.encode('latin-1')
                
                file_buffer = io.BytesIO()
                file_buffer.write(pdf_data)
                file_buffer.seek(0)
            
            return send_file(
                file_buffer,
                as_attachment=True,
                download_name='corrected_cv.pdf',
                mimetype='application/pdf'
            )
        except Exception as e:
            flash(f'Error generating PDF: {str(e)}', 'danger')
            logging.error(f"PDF generation error: {str(e)}")
            return redirect(url_for('results'))

@app.route('/apply_corrections', methods=['POST'])
def apply_corrections():
    if request.method == 'POST':
        data = request.json
        if not data or 'selected_corrections' not in data or 'resume_text' not in data:
            return json.dumps({'error': 'Invalid data format'}), 400, {'ContentType': 'application/json'}
        
        selected_corrections = data['selected_corrections']
        resume_text = data['resume_text']
        
        # Apply selected corrections to the CV text
        # Sort corrections in reverse order to avoid offset issues
        sorted_corrections = sorted(selected_corrections, key=lambda x: x['position']['start'], reverse=True)
        
        for correction in sorted_corrections:
            start = correction['position']['start']
            end = correction['position']['end']
            resume_text = resume_text[:start] + correction['suggestion'] + resume_text[end:]
        
        return json.dumps({'corrected_text': resume_text}), 200, {'ContentType': 'application/json'}
    
    return json.dumps({'error': 'Invalid request method'}), 405, {'ContentType': 'application/json'}


@app.route('/anschreiben', methods=['GET', 'POST'])
def anschreiben_page():
    if request.method == 'GET':
        # Check if CV is in session
        if 'resume_text' not in session:
            flash('Please submit a CV for analysis first.', 'warning')
            return redirect(url_for('index'))
            
        # Get CV from session
        resume_text = session.get('resume_text', '')
        anschreiben_text = session.get('anschreiben', '')
        job_description = session.get('job_description', '')
        
        return render_template('anschreiben.html', 
                              resume_text=resume_text,
                              job_description=job_description,
                              anschreiben=anschreiben_text)
    
    elif request.method == 'POST':
        # Get CV and job description
        resume_text = session.get('resume_text', '')
        job_description = request.form.get('job_description', '')
        
        if not resume_text:
            flash('No CV found. Please upload your CV first.', 'warning')
            return redirect(url_for('index'))
            
        if not job_description or len(job_description) < 50:
            flash('Please provide a detailed job description to generate a personalized cover letter.', 'warning')
            return render_template('anschreiben.html', 
                                  resume_text=resume_text,
                                  job_description=job_description,
                                  anschreiben='')
        
        try:
            # Generate anschreiben using DeepInfra API
            anschreiben_text = generate_anschreiben(resume_text, job_description)
            
            # Store in session
            session['anschreiben'] = anschreiben_text
            session['job_description'] = job_description
            
            return render_template('anschreiben.html', 
                                  resume_text=resume_text,
                                  job_description=job_description,
                                  anschreiben=anschreiben_text)
                                  
        except Exception as e:
            error_msg = str(e)
            # Add a more user-friendly message for API errors
            if "API error: 401" in error_msg:
                flash('Authentication error with DeepInfra API. Using template-based generation instead.', 'warning')
            else:
                flash(f'Error generating Anschreiben: {error_msg}', 'danger')
                
            logging.error(f"Anschreiben generation error: {error_msg}")
            logging.error(traceback.format_exc())
            
            return render_template('anschreiben.html', 
                                  resume_text=resume_text,
                                  job_description=job_description,
                                  anschreiben='')


@app.route('/donate')
def donate():
    """Display the donation page."""
    return render_template('donate.html')


@app.route('/download_anschreiben', methods=['POST'])
def download_anschreiben():
    if 'anschreiben' not in session:
        flash('No Anschreiben available for download.', 'warning')
        return redirect(url_for('anschreiben_page'))
    
    # Get the anschreiben text from the session
    anschreiben_text = session.get('anschreiben', '')
    
    if not anschreiben_text:
        flash('No content to download.', 'warning')
        return redirect(url_for('anschreiben_page'))
    
    # Create a temporary file with the anschreiben
    format_type = request.form.get('format', 'txt')
    
    if format_type == 'txt':
        # Create a text file
        file_buffer = io.BytesIO()
        file_buffer.write(anschreiben_text.encode('utf-8'))
        file_buffer.seek(0)
        
        return send_file(
            file_buffer,
            as_attachment=True,
            download_name='anschreiben.txt',
            mimetype='text/plain'
        )
    else:  # PDF
        try:
            from fpdf import FPDF
            
            # Create a PDF file
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            # Clean text of non-compatible characters
            clean_text = anschreiben_text.replace('\u2013', '-')  # en-dash
            clean_text = clean_text.replace('\u2014', '--')     # em-dash
            clean_text = clean_text.replace('\u2018', "'")      # left single quote
            clean_text = clean_text.replace('\u2019', "'")      # right single quote
            clean_text = clean_text.replace('\u201c', '"')      # left double quote
            clean_text = clean_text.replace('\u201d', '"')      # right double quote
            
            # Split the text into lines and add to PDF
            for line in clean_text.split('\n'):
                # Handle potential Unicode errors with a general filter
                safe_line = ''.join(c if ord(c) < 128 else '-' for c in line)
                pdf.multi_cell(0, 10, safe_line)
            
            # Save the PDF to a memory buffer
            file_buffer = io.BytesIO()
            
            try:
                # Get the PDF as bytes
                pdf_data = pdf.output(dest='S')
                if isinstance(pdf_data, str):
                    pdf_data = pdf_data.encode('latin-1')
                
                file_buffer.write(pdf_data)
                file_buffer.seek(0)
            except UnicodeEncodeError:
                logging.warning("Unicode encode error when creating PDF, using ASCII fallback")
                # If still encountering unicode errors, fall back to ASCII only
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                
                # Use ASCII-only version
                ascii_text = anschreiben_text.encode('ascii', 'replace').decode('ascii')
                for line in ascii_text.split('\n'):
                    pdf.multi_cell(0, 10, line)
                
                pdf_data = pdf.output(dest='S')
                if isinstance(pdf_data, str):
                    pdf_data = pdf_data.encode('latin-1')
                
                file_buffer = io.BytesIO()
                file_buffer.write(pdf_data)
                file_buffer.seek(0)
            
            return send_file(
                file_buffer,
                as_attachment=True,
                download_name='anschreiben.pdf',
                mimetype='application/pdf'
            )
        except Exception as e:
            flash(f'Error generating PDF: {str(e)}', 'danger')
            logging.error(f"PDF generation error: {str(e)}")
            return redirect(url_for('anschreiben_page'))
