import os
import io
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from werkzeug.utils import secure_filename
import tempfile
import json
from resume_parser import parse_resume_file, parse_resume_text
from ai_analyzer import analyze_resume

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

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
    # Clear any previous resume data from session
    if 'resume_text' in session:
        session.pop('resume_text')
    if 'corrections' in session:
        session.pop('corrections')
    
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
            
            # Parse the resume file
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
        flash('Please upload a file or paste your resume text.', 'warning')
        return redirect(url_for('index'))
    
    # Check if resume text is too short
    if len(resume_text) < 50:
        flash('The resume content is too short or could not be properly extracted. Please try again.', 'warning')
        return redirect(url_for('index'))
    
    try:
        # Analyze the resume with OpenAI
        corrections = analyze_resume(resume_text)
        
        # Store the resume and corrections in the session
        session['resume_text'] = resume_text
        session['corrections'] = corrections
        
        return redirect(url_for('results'))
        
    except Exception as e:
        flash(f'Error analyzing resume: {str(e)}', 'danger')
        logging.error(f"AI analysis error: {str(e)}")
        return redirect(url_for('index'))

@app.route('/results')
def results():
    # Check if resume and corrections are in session
    if 'resume_text' not in session or 'corrections' not in session:
        flash('Please submit a resume for analysis first.', 'warning')
        return redirect(url_for('index'))
    
    resume_text = session['resume_text']
    corrections = session['corrections']
    
    return render_template('result.html', resume_text=resume_text, corrections=corrections)

@app.route('/download', methods=['POST'])
def download():
    if 'resume_text' not in session:
        flash('No resume data available for download.', 'warning')
        return redirect(url_for('index'))
    
    # Get the corrected resume text from the form
    corrected_text = request.form.get('corrected_text', '')
    
    if not corrected_text:
        flash('No content to download.', 'warning')
        return redirect(url_for('results'))
    
    # Create a temporary file with the corrected resume
    format_type = request.form.get('format', 'txt')
    
    if format_type == 'txt':
        # Create a text file
        file_buffer = io.BytesIO()
        file_buffer.write(corrected_text.encode('utf-8'))
        file_buffer.seek(0)
        
        return send_file(
            file_buffer,
            as_attachment=True,
            download_name='corrected_resume.txt',
            mimetype='text/plain'
        )
    else:  # PDF
        try:
            from fpdf import FPDF
            
            # Create a PDF file
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            # Split the text into lines and add to PDF
            for line in corrected_text.split('\n'):
                pdf.multi_cell(0, 10, line)
            
            # Save the PDF to a memory buffer
            file_buffer = io.BytesIO()
            
            # Get the PDF as a string and convert to bytes
            pdf_data = pdf.output(dest='S')
            if isinstance(pdf_data, str):
                pdf_data = pdf_data.encode('latin-1')
            
            file_buffer.write(pdf_data)
            file_buffer.seek(0)
            
            return send_file(
                file_buffer,
                as_attachment=True,
                download_name='corrected_resume.pdf',
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
        
        # Apply selected corrections to the resume text
        # Sort corrections in reverse order to avoid offset issues
        sorted_corrections = sorted(selected_corrections, key=lambda x: x['position']['start'], reverse=True)
        
        for correction in sorted_corrections:
            start = correction['position']['start']
            end = correction['position']['end']
            resume_text = resume_text[:start] + correction['suggestion'] + resume_text[end:]
        
        return json.dumps({'corrected_text': resume_text}), 200, {'ContentType': 'application/json'}
    
    return json.dumps({'error': 'Invalid request method'}), 405, {'ContentType': 'application/json'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
