import os
import json
import logging
import re
import requests

# Hugging Face API configuration
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/prithivida/grammar_error_correcter_v1"
HUGGINGFACE_TEXT_GEN_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")

# Headers for Hugging Face API request
api_headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
    "Content-Type": "application/json"
}

def analyze_resume(resume_text):
    """
    Analyze resume text using Hugging Face API to provide improvement suggestions.
    
    Args:
        resume_text (str): The resume text to analyze
        
    Returns:
        list: A list of correction suggestions with their positions and explanations
    """
    try:
        # Split the resume into sentences/paragraphs for better analysis
        segments = split_text_into_segments(resume_text)
        corrections = []
        
        # Process each segment
        for segment in segments:
            # Skip empty segments
            if not segment.strip():
                continue
                
            segment_corrections = analyze_segment(segment, resume_text)
            corrections.extend(segment_corrections)
        
        # If no corrections were found, perform a fallback analysis for common issues
        if not corrections:
            fallback_corrections = perform_fallback_analysis(resume_text)
            corrections.extend(fallback_corrections)
            
        return corrections
        
    except Exception as e:
        logging.error(f"Error analyzing resume: {str(e)}")
        raise Exception(f"Error analyzing resume: {str(e)}")

def split_text_into_segments(text, max_segment_length=200):
    """
    Split text into manageable segments for API processing.
    
    Args:
        text (str): The text to split
        max_segment_length (int): Maximum length of each segment
        
    Returns:
        list: List of text segments
    """
    # First try to split by paragraphs
    paragraphs = text.split('\n')
    segments = []
    
    for paragraph in paragraphs:
        # If paragraph is short enough, add it directly
        if len(paragraph) <= max_segment_length:
            segments.append(paragraph)
        else:
            # Split longer paragraphs into sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            current_segment = ""
            
            for sentence in sentences:
                if len(current_segment) + len(sentence) <= max_segment_length:
                    current_segment += " " + sentence if current_segment else sentence
                else:
                    if current_segment:
                        segments.append(current_segment)
                    # If a single sentence is too long, split it by length
                    if len(sentence) > max_segment_length:
                        for i in range(0, len(sentence), max_segment_length):
                            segments.append(sentence[i:i + max_segment_length])
                    else:
                        current_segment = sentence
            
            if current_segment:
                segments.append(current_segment)
    
    return segments

def analyze_segment(segment, full_text):
    """
    Analyze a segment of text using the Hugging Face API.
    
    Args:
        segment (str): The text segment to analyze
        full_text (str): The full resume text (for position calculation)
        
    Returns:
        list: Corrections for this segment
    """
    corrections = []
    
    try:
        # Call Hugging Face API
        payload = {"inputs": segment}
        response = requests.post(HUGGINGFACE_API_URL, headers=api_headers, json=payload)
        response.raise_for_status()
        
        # Process API response
        corrected_text = response.json()[0]["generated_text"]
        
        # Регулярное выражение для игнорирования изменений в точках, 
        # где модель меняет только точку или пробел + точку
        import re
        segment_normalized = re.sub(r'\s*\.', '.', segment)
        corrected_normalized = re.sub(r'\s*\.', '.', corrected_text)
        
        # If there's a difference between original and corrected text
        # except for just period-related changes
        if segment_normalized != corrected_normalized and segment != corrected_text:
            # Check if the only difference is periods/dots
            # Remove dots and compare to avoid false corrections for dots
            segment_no_dots = segment.replace('.', '')
            corrected_no_dots = corrected_text.replace('.', '')
            
            # If they're identical after removing dots, don't create a correction
            if segment_no_dots == corrected_no_dots:
                return corrections
                
            # Find the position in the full text
            start_pos = full_text.find(segment)
            end_pos = start_pos + len(segment)
            
            # Create correction object
            correction = {
                "original": segment,
                "position": {"start": start_pos, "end": end_pos},
                "suggestion": corrected_text,
                "explanation": "Grammar and spelling improvement",
                "category": "grammar"
            }
            
            corrections.append(correction)
            
    except requests.exceptions.RequestException as e:
        logging.warning(f"Hugging Face API request failed: {str(e)}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logging.warning(f"Error parsing Hugging Face API response: {str(e)}")
    except Exception as e:
        logging.warning(f"Unexpected error in segment analysis: {str(e)}")
    
    return corrections

def perform_fallback_analysis(text):
    """
    Perform basic text analysis for common resume issues when the API fails.
    
    Args:
        text (str): The resume text to analyze
        
    Returns:
        list: List of corrections
    """
    corrections = []
    
    # Common resume improvements to check for
    checks = [
        {
            "pattern": r"\bresponsible for\b",
            "replacement": "managed",
            "explanation": "Use action verbs instead of passive phrases",
            "category": "clarity"
        },
        {
            "pattern": r"\bhelped\b",
            "replacement": "assisted",
            "explanation": "Use more professional terminology",
            "category": "professional language"
        },
        {
            "pattern": r"\bi\b",
            "replacement": "",
            "explanation": "Avoid using first-person pronouns in resumes",
            "category": "professional language"
        },
        {
            "pattern": r"\bteam player\b",
            "replacement": "collaborative professional",
            "explanation": "Avoid overused phrases and clichés",
            "category": "content"
        },
        {
            "pattern": r"\bms office\b",
            "replacement": "Microsoft Office",
            "explanation": "Use proper capitalization for product names",
            "category": "formatting"
        },
    ]
    
    for check in checks:
        for match in re.finditer(check["pattern"], text, re.IGNORECASE):
            start_pos = match.start()
            end_pos = match.end()
            original_text = text[start_pos:end_pos]
            
            # Create replacement based on case of original
            if original_text.isupper():
                replacement = check["replacement"].upper()
            elif original_text[0].isupper():
                replacement = check["replacement"].capitalize()
            else:
                replacement = check["replacement"]
                
            # Skip if replacement would be empty
            if not replacement:
                continue
                
            correction = {
                "original": original_text,
                "position": {"start": start_pos, "end": end_pos},
                "suggestion": replacement,
                "explanation": check["explanation"],
                "category": check["category"]
            }
            
            corrections.append(correction)
    
    return corrections


def generate_anschreiben(resume_text, job_description):
    """
    Generate a personalized cover letter (Anschreiben) based on resume and job description.
    
    Args:
        resume_text (str): The resume text to analyze
        job_description (str): The job description text
        
    Returns:
        str: Generated cover letter text
    """
    try:
        # Extract skills and experiences from resume
        skills = extract_skills_from_resume(resume_text)
        
        # Create prompt for Hugging Face API
        prompt = f"""<s>[INST] Du bist ein professioneller Bewerbungsexperte. Erstelle ein Anschreiben auf Deutsch, basierend auf den folgenden Informationen aus dem Lebenslauf und der Stellenanzeige.

Lebenslauf:
{resume_text}

Stellenanzeige:
{job_description}

Verwende die folgenden Anweisungen:
1. Das Anschreiben sollte formell und professionell sein
2. Betone die Übereinstimmung der Qualifikationen mit den Anforderungen
3. Halte es präzise und auf ca. 300-350 Wörter beschränkt
4. Verwende folgende Struktur:
   - Anrede (falls in der Stellenanzeige ein Name genannt wird)
   - Einleitung mit Bezug auf die Stelle
   - Hauptteil mit Bezug auf den Lebenslauf und passenden Qualifikationen
   - Motivation für die Stelle
   - Abschluss mit Gesprächsbereitschaft
   - Grußformel

Achte auf eine professionelle Sprache und einen überzeugenden Stil.
[/INST]</s>
"""
        
        # Call Hugging Face API for text generation
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
        }
        
        response = requests.post(
            HUGGINGFACE_TEXT_GEN_API_URL,
            headers=api_headers,
            json=payload
        )
        
        response.raise_for_status()
        
        # Extract the generated text from the response
        result = response.json()
        
        # Parse and clean the generated anschreiben
        anschreiben = parse_anschreiben_from_response(result)
        
        return anschreiben
        
    except Exception as e:
        logging.error(f"Error generating Anschreiben: {str(e)}")
        # Если API не работает, используем шаблонный подход как запасной вариант
        try:
            logging.info("Using template-based approach as fallback")
            skills_info = extract_skills_from_resume(resume_text)
            job_title = extract_job_title(job_description)
            company_name = extract_company_name(job_description)
            return generate_template_anschreiben(resume_text, job_description, job_title, company_name, skills_info)
        except Exception as template_err:
            logging.error(f"Error in template fallback: {str(template_err)}")
            raise Exception(f"Error generating Anschreiben: {str(e)}")


def extract_skills_from_resume(resume_text):
    """
    Extract skills and key information from resume text.
    
    Args:
        resume_text (str): The resume text to analyze
        
    Returns:
        dict: Extracted skills and information
    """
    skills = {
        "technical_skills": [],
        "languages": [],
        "education": [],
        "experience": []
    }
    
    # Extract languages (simplified approach)
    language_patterns = [
        r"\b(?:Deutsch|Englisch|Französisch|Spanisch|Italienisch|Russisch|Chinesisch|Japanisch)\b(?:\s+\((?:Muttersprache|[BCG][12]|fließend|verhandlungssicher|Grundkenntnisse)\))?",
        r"\b(?:German|English|French|Spanish|Italian|Russian|Chinese|Japanese)\b(?:\s+\((?:native|[BCG][12]|fluent|business|basic)\))?"
    ]
    
    for pattern in language_patterns:
        matches = re.findall(pattern, resume_text, re.IGNORECASE)
        skills["languages"].extend(matches)
    
    # Extract technical skills (simplified)
    tech_patterns = [
        r"\b(?:Java|Python|C\+\+|JavaScript|SQL|PHP|HTML|CSS|React|Angular|Vue|Node\.js|Excel|Word|PowerPoint|SAP)\b"
    ]
    
    for pattern in tech_patterns:
        matches = re.findall(pattern, resume_text, re.IGNORECASE)
        skills["technical_skills"].extend(matches)
    
    # Extract education information (simplified)
    edu_patterns = [
        r"(?:Universität|Hochschule|Fachhochschule|University|College|Institute)[^\n.]{3,50}",
        r"(?:Bachelor|Master|Diplom|Promotion|PhD|Dr\.)[^\n.]{3,50}"
    ]
    
    for pattern in edu_patterns:
        matches = re.findall(pattern, resume_text, re.IGNORECASE)
        skills["education"].extend(matches)
    
    # Extract job experiences (simplified)
    exp_patterns = [
        r"(?:Software Engineer|Developer|Entwickler|Projektmanager|Manager|Consultant|Berater)[^\n.]{3,50}"
    ]
    
    for pattern in exp_patterns:
        matches = re.findall(pattern, resume_text, re.IGNORECASE)
        skills["experience"].extend(matches)
    
    # Remove duplicates
    skills["languages"] = list(set(skills["languages"]))
    skills["technical_skills"] = list(set(skills["technical_skills"]))
    skills["education"] = list(set(skills["education"]))
    skills["experience"] = list(set(skills["experience"]))
    
    return skills


def parse_anschreiben_from_response(api_response):
    """
    Parse and clean the Anschreiben from the API response.
    
    Args:
        api_response: The API response to parse
        
    Returns:
        str: Cleaned Anschreiben text
    """
    try:
        # Extract generated text from model response
        if isinstance(api_response, list) and len(api_response) > 0:
            generated_text = api_response[0].get("generated_text", "")
        elif isinstance(api_response, dict):
            generated_text = api_response.get("generated_text", "")
        else:
            generated_text = str(api_response)
        
        # Extract the content after [/INST]
        if "[/INST]" in generated_text:
            anschreiben = generated_text.split("[/INST]")[1].strip()
        else:
            anschreiben = generated_text.strip()
        
        # Remove any trailing model tokens or artifacts
        anschreiben = re.sub(r'<\/s>$', '', anschreiben).strip()
        
        return anschreiben
        
    except Exception as e:
        logging.error(f"Error parsing Anschreiben response: {str(e)}")
        return "Error generating Anschreiben. Please try again."


def extract_job_title(job_description):
    """
    Extract job title from job description.
    
    Args:
        job_description (str): The job description text
        
    Returns:
        str: Extracted job title
    """
    # Try to find common job title patterns
    title_patterns = [
        r"(?:Stellenanzeige|Stelle|Position|Job)[:\s]+([^\n.]{5,50})",
        r"(?:Wir suchen|Gesucht)[:\s]+([^\n.]{5,50})",
        r"^([^\n.]{5,50})(?:\n|$)"
    ]
    
    for pattern in title_patterns:
        matches = re.findall(pattern, job_description, re.IGNORECASE)
        if matches:
            return matches[0].strip()
    
    # Default title if no match found
    return "die ausgeschriebene Stelle"


def extract_company_name(job_description):
    """
    Extract company name from job description.
    
    Args:
        job_description (str): The job description text
        
    Returns:
        str: Extracted company name
    """
    # Try to find common company name patterns
    company_patterns = [
        r"(?:Firma|Unternehmen|Company)[:\s]+([^\n.]{2,30})",
        r"(?:bei der|bei|at|für die|für)\s+([A-Z][^\n.]{2,30})\s+(?:GmbH|AG|SE|KG|OHG|LLC|Inc|Ltd)"
    ]
    
    for pattern in company_patterns:
        matches = re.findall(pattern, job_description, re.IGNORECASE)
        if matches:
            return matches[0].strip()
    
    # Default if no match found
    return "Ihrem Unternehmen"


def generate_template_anschreiben(resume_text, job_description, job_title, company_name, skills_info):
    """
    Generate Anschreiben using a template-based approach.
    
    Args:
        resume_text (str): The resume text
        job_description (str): The job description text
        job_title (str): The extracted job title
        company_name (str): The extracted company name
        skills_info (dict): Extracted skills and information
        
    Returns:
        str: Generated Anschreiben text
    """
    # Get current date in German format
    from datetime import datetime
    current_date = datetime.now().strftime("%d.%m.%Y")
    
    # Get skills as comma-separated lists
    tech_skills = ", ".join(skills_info["technical_skills"][:5]) if skills_info["technical_skills"] else "verschiedene technische Fähigkeiten"
    languages = ", ".join(skills_info["languages"][:3]) if skills_info["languages"] else "verschiedene Sprachen"
    
    # Get the most recent experience
    experience = skills_info["experience"][0] if skills_info["experience"] else "meine bisherige Berufserfahrung"
    
    # Template for Anschreiben
    anschreiben_template = f"""
{current_date}

Betreff: Bewerbung als {job_title}

Sehr geehrte Damen und Herren,

mit großem Interesse habe ich Ihre Stellenanzeige für die Position als {job_title} bei {company_name} gelesen. Aufgrund meiner Qualifikationen und Erfahrungen bin ich überzeugt, dass ich eine wertvolle Ergänzung für Ihr Team sein kann.

Wie Sie meinem Lebenslauf entnehmen können, verfüge ich über Erfahrung als {experience}. Während meiner beruflichen Laufbahn konnte ich meine Fähigkeiten in {tech_skills} kontinuierlich verbessern und erfolgreich in verschiedenen Projekten anwenden. Zudem kann ich in {languages} kommunizieren, was die Zusammenarbeit in internationalen Teams erleichtert.

Die in Ihrer Stellenbeschreibung genannten Anforderungen entsprechen genau meinem Profil und meinen beruflichen Zielen. Besonders reizt mich die Möglichkeit, meine Kenntnisse bei {company_name} einzubringen und weiterzuentwickeln.

Ich freue mich auf die Gelegenheit, meine Bewerbung in einem persönlichen Gespräch zu vertiefen und mehr über die Position zu erfahren. Für Rückfragen stehe ich Ihnen jederzeit gerne zur Verfügung.

Mit freundlichen Grüßen,

[Ihr Name]
"""
    
    return anschreiben_template.strip()
