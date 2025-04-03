import logging
import os

def parse_resume_file(file_path):
    """
    Parse a resume file (PDF, DOCX, or TXT) and extract the text.
    
    Args:
        file_path (str): Path to the resume file
        
    Returns:
        str: Extracted text from the resume
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return parse_pdf(file_path)
    elif file_extension == '.docx':
        return parse_docx(file_path)
    elif file_extension == '.txt':
        return parse_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def parse_pdf(file_path):
    """
    Parse a PDF file and extract the text.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(file_path)
        return text
    except ImportError:
        logging.error("pdfminer.six is not installed. Unable to parse PDF files.")
        raise ImportError("pdfminer.six is not installed. Unable to parse PDF files.")
    except Exception as e:
        logging.error(f"Error parsing PDF file: {str(e)}")
        raise Exception(f"Error parsing PDF file: {str(e)}")

def parse_docx(file_path):
    """
    Parse a DOCX file and extract the text.
    
    Args:
        file_path (str): Path to the DOCX file
        
    Returns:
        str: Extracted text from the DOCX
    """
    try:
        import docx
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except ImportError:
        logging.error("python-docx is not installed. Unable to parse DOCX files.")
        raise ImportError("python-docx is not installed. Unable to parse DOCX files.")
    except Exception as e:
        logging.error(f"Error parsing DOCX file: {str(e)}")
        raise Exception(f"Error parsing DOCX file: {str(e)}")

def parse_txt(file_path):
    """
    Parse a TXT file and extract the text.
    
    Args:
        file_path (str): Path to the TXT file
        
    Returns:
        str: Extracted text from the TXT
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        logging.error(f"Error parsing TXT file: {str(e)}")
        raise Exception(f"Error parsing TXT file: {str(e)}")

def parse_resume_text(text):
    """
    Process raw resume text input.
    
    Args:
        text (str): Raw resume text
        
    Returns:
        str: Processed resume text
    """
    # Remove excessive whitespace
    text = " ".join(text.split())
    # Replace multiple newlines with single newlines
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text
