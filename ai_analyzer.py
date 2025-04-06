import os
import json
import logging
import re
import requests
import traceback
from datetime import datetime

# Groq API configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
GROQ_MODEL = "llama3-70b-8192"  # Using Llama-3 70B model

# Initialize Groq client
groq_client = None
try:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)
    logging.info("Groq client initialized successfully")
except ImportError:
    logging.error("Failed to import Groq client. Make sure the groq package is installed.")
except Exception as e:
    logging.error(f"Error initializing Groq client: {str(e)}")

# For backward compatibility with older API code
DEEPINFRA_API_KEY = GROQ_API_KEY
DEEPINFRA_MODEL = "meta-llama/Llama-3-8b-instruct"
DEEPINFRA_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Headers for API requests
deepinfra_headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# Keep Hugging Face API for grammar correction (backward compatibility)
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/prithivida/grammar_error_correcter_v1"
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")

# Headers for Hugging Face API request
huggingface_headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
    "Content-Type": "application/json"
}

def analyze_resume(resume_text, language='en'):
    """
    Analyze CV text using AI to provide improvement suggestions.
    
    Args:
        resume_text (str): The CV text to analyze
        language (str): Language for the analysis ('en' or 'de')
        
    Returns:
        list: A list of correction suggestions with their positions and explanations
    """
    try:
        # First try to use DeepInfra API for comprehensive analysis
        if DEEPINFRA_API_KEY:
            try:
                logging.info("Attempting to use DeepInfra API for CV analysis")
                corrections = analyze_with_deepinfra(resume_text, language)
                if corrections:
                    logging.info(f"DeepInfra API analysis successful, found {len(corrections)} suggestions")
                    return corrections
                logging.warning("DeepInfra API returned no corrections")
            except Exception as api_error:
                logging.error(f"DeepInfra API analysis error: {str(api_error)}")
                
        # Fallback to rules-based analysis
        logging.info("Using enhanced fallback analysis for CV corrections")
        fallback_corrections = perform_enhanced_analysis(resume_text, language)
        return fallback_corrections
    
    except Exception as e:
        logging.error(f"Error analyzing CV: {str(e)}")
        logging.error(f"Stack trace: {traceback.format_exc()}")
        
        # Instead of raising the exception, return basic fallback corrections
        logging.info("Using basic fallback analysis due to error.")
        return perform_fallback_analysis(resume_text)

def split_text_into_segments(text, max_segment_length=200):
    """
    Split text into manageable segments for API processing.
    
    Args:
        text (str): The text to split
        max_segment_length (int): Maximum length of each segment
        
    Returns:
        list: List of text segments
    """
    # Simple splitting by sentences or newlines
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    
    segments = []
    current_segment = ""
    
    for sentence in sentences:
        # Skip empty sentences
        if not sentence.strip():
            continue
            
        # If adding this sentence would exceed max length, start a new segment
        if len(current_segment) + len(sentence) > max_segment_length and current_segment:
            segments.append(current_segment.strip())
            current_segment = sentence
        else:
            # Add space only if current_segment is not empty
            if current_segment:
                current_segment += " "
            current_segment += sentence
    
    # Add the last segment
    if current_segment:
        segments.append(current_segment.strip())
    
    return segments

def analyze_segment(segment, full_text):
    """
    Analyze a segment of text using the Hugging Face API.
    
    Args:
        segment (str): The text segment to analyze
        full_text (str): The full CV text (for position calculation)
        
    Returns:
        list: Corrections for this segment
    """
    corrections = []
    
    try:
        # Check if API key is present
        if not HUGGINGFACE_API_KEY:
            logging.error("No Hugging Face API key provided")
            return []
            
        # Prepare payload
        payload = json.dumps({"inputs": segment})
        
        # Make request to Hugging Face API
        response = requests.post(
            HUGGINGFACE_API_URL,
            headers=huggingface_headers,
            data=payload,
            timeout=10  # 10 second timeout
        )
        
        # Check for successful response
        if response.status_code == 200:
            response_json = response.json()
            
            # Process the response
            if isinstance(response_json, list) and len(response_json) > 0:
                original_text = segment
                corrected_text = response_json[0].get("generated_text", "")
                
                # Skip if API returns the same text
                if original_text == corrected_text:
                    return []
                
                # Find the position of segment in full text
                segment_start = full_text.find(segment)
                if segment_start == -1:
                    # If exact match not found, try case-insensitive
                    pattern = re.escape(segment)
                    matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
                    if matches:
                        segment_start = matches[0].start()
                    else:
                        # Skip if segment can't be located
                        logging.warning(f"Segment not found in full text: {segment[:30]}...")
                        return []
                        
                # Create a correction
                correction = {
                    "original": original_text,
                    "position": {"start": segment_start, "end": segment_start + len(original_text)},
                    "suggestion": corrected_text,
                    "explanation": "Improved grammar and clarity",
                    "category": "grammar"
                }
                
                corrections.append(correction)
        else:
            logging.warning(f"Hugging Face API error: {response.status_code}, {response.text}")
            
    except requests.exceptions.RequestException as e:
        logging.warning(f"Hugging Face API request failed: {str(e)}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logging.warning(f"Error parsing Hugging Face API response: {str(e)}")
    except Exception as e:
        logging.warning(f"Unexpected error in segment analysis: {str(e)}")
        logging.error(f"Exception traceback: {traceback.format_exc()}")
    
    return corrections

def perform_fallback_analysis(text):
    """
    Perform basic text analysis for common CV issues when the API fails.
    
    Args:
        text (str): The CV text to analyze
        
    Returns:
        list: List of corrections
    """
    corrections = []
    
    # Common CV improvements to check for
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
            "explanation": "Avoid using first-person pronouns in CVs",
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
        {
            "pattern": r"\b(?:gute|sehr gute|ausgezeichnete)\s+kenntnisse\b",
            "replacement": "Fortgeschrittene Kenntnisse",
            "explanation": "Be more specific and professional in describing skills",
            "category": "professional language"
        },
        {
            "pattern": r"\bargts", 
            "replacement": "arbeitet",
            "explanation": "Use full words instead of abbreviations",
            "category": "professional language"
        },
        {
            "pattern": r"\binteragirt\b",
            "replacement": "interagiert",
            "explanation": "Fix spelling errors",
            "category": "spelling"
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
    Generate a personalized cover letter (Anschreiben) based on CV and job description.
    
    Args:
        resume_text (str): The CV text to analyze
        job_description (str): The job description text
        
    Returns:
        str: Generated cover letter text
    """
    try:
        # Extract key info from resume and job description
        skills_info = extract_skills_from_resume(resume_text)
        job_title = extract_job_title(job_description)
        company_name = extract_company_name(job_description)
        
        # First try using Groq
        if groq_client:
            try:
                logging.info("Using Groq to generate Anschreiben")
                
                system_prompt = """You are an expert in writing professional cover letters (Anschreiben) for 
                German job applications. Create a formal, professional cover letter that matches the applicant's 
                qualifications to the job requirements. Follow German business letter standards."""
                
                user_prompt = f"""Create a personalized cover letter (Anschreiben) in German based on the following information:
                
                CV Highlights:
                Technical Skills: {', '.join(skills_info['technical_skills'][:5])}
                Languages: {', '.join(skills_info['languages'][:3])}
                Education: {', '.join(skills_info['education'][:2])}
                Experience: {', '.join(skills_info['experience'][:3])}
                
                Job Position: {job_title}
                Company: {company_name}
                
                Job Description:
                {job_description[:500]}
                
                Format as a proper German business letter with all standard sections.
                """
                
                # Make API call to Groq
                response = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                # Extract the generated content
                if response and response.choices and len(response.choices) > 0:
                    logging.info("Groq successfully generated Anschreiben")
                    anschreiben_text = response.choices[0].message.content
                    return anschreiben_text
                else:
                    logging.warning("Empty response from Groq API")
                    
            except Exception as groq_error:
                logging.error(f"Groq API error: {str(groq_error)}")
                logging.error(f"Will fall back to template-based generation")
        
        # If Groq fails or not available, fall back to template
        logging.info("Using template-based approach for Anschreiben generation")
        return generate_template_anschreiben(resume_text, job_description, job_title, company_name, skills_info)
        
    except Exception as e:
        logging.error(f"Error generating Anschreiben: {str(e)}")
        logging.error(f"Stack trace: {traceback.format_exc()}")
        
        # Fall back to template-based generation
        try:
            logging.info("Using template-based approach as fallback")
            skills_info = extract_skills_from_resume(resume_text)
            job_title = extract_job_title(job_description)
            company_name = extract_company_name(job_description)
            return generate_template_anschreiben(resume_text, job_description, job_title, company_name, skills_info)
        except Exception as template_err:
            logging.error(f"Error in template fallback: {str(template_err)}")
            logging.error(f"Template fallback stack trace: {traceback.format_exc()}")
            return "Fehler bei der Erstellung des Anschreibens. Bitte versuchen Sie es später erneut."


def extract_skills_from_resume(resume_text):
    """
    Extract skills and key information from CV text.
    
    Args:
        resume_text (str): The CV text to analyze
        
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
        # API response could be in different formats depending on provider
        # For DeepInfra, we already extract the content directly in the calling function
        
        # If it's a string already, just use it
        if isinstance(api_response, str):
            return api_response.strip()
            
        # If it's a list from Hugging Face API
        if isinstance(api_response, list) and len(api_response) > 0:
            generated_text = api_response[0].get("generated_text", "")
        # If it's a dict
        elif isinstance(api_response, dict):
            # DeepInfra format
            if "choices" in api_response and len(api_response["choices"]) > 0:
                if "message" in api_response["choices"][0]:
                    generated_text = api_response["choices"][0]["message"].get("content", "")
                else:
                    generated_text = api_response["choices"][0].get("text", "")
            # Hugging Face format
            else:
                generated_text = api_response.get("generated_text", "")
        else:
            generated_text = str(api_response)
        
        # Extract the content after [/INST] if present
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


def score_resume(resume_text, language='en'):
    """
    Score CV on a scale from 1-100 based on German standards.
    
    Args:
        resume_text (str): The CV text to analyze
        language (str): Language for the summary and tips ('en' or 'de')
        
    Returns:
        dict: Score details including overall score and category scores
    """
    try:
        # Try using DeepInfra API for scoring
        return score_resume_with_api(resume_text, language)
    except Exception as e:
        logging.error(f"API-based scoring failed: {str(e)}")
        logging.error(traceback.format_exc())
        
        # Fallback to rule-based scoring
        scores = {}
        
        # Content score (40%)
        content_score = 0
        
        # Check for achievements with numbers
        achievement_pattern = r"\b(?:increased|improved|reduced|saved|achieved|developed|launched|managed|led)\b.*?\b\d+(?:\.\d+)?%?\b"
        achievements = re.findall(achievement_pattern, resume_text, re.IGNORECASE)
        content_score += min(len(achievements) * 5, 20)  # 5 points per achievement, max 20
        
        # Check for skills section
        if re.search(r"\b(?:skills|fähigkeiten|kenntnisse|kompetenzen)\b", resume_text, re.IGNORECASE):
            content_score += 10
            
        # Check for education details
        if re.search(r"\b(?:bachelor|master|diplom|doktor|phd|ausbildung|studium|university|hochschule|universität)\b", resume_text, re.IGNORECASE):
            content_score += 10
        
        # Format score (30%)
        format_score = 0
        
        # Check for bullet points
        bullet_points = re.findall(r"(?:^|\n)\s*(?:•|-|\*|\d+\.)\s+", resume_text)
        format_score += min(len(bullet_points) * 2, 15)  # 2 points per bullet, max 15
        
        # Check for sections
        section_headers = re.findall(r"(?:^|\n)(?:[A-Z][A-Za-z\s]+:|\b(?:EDUCATION|EXPERIENCE|SKILLS|PROJECTS|AUSBILDUNG|BERUFSERFAHRUNG|FÄHIGKEITEN|PROJEKTE)\b)", resume_text)
        format_score += min(len(section_headers) * 3, 15)  # 3 points per section, max 15
        
        # Language score (15%)
        language_score = 0
        
        # Check for grammatical errors (simplistic)
        grammar_patterns = [
            r"\bi\s+(?:has|have|is|am|are|was|were)\b",  # Subject-verb agreement issues
            r"\b(?:a|an)\s+(?:[aeiou]|hour|honor)",       # A/An issues
            r"\b(?:he|she|it|they)\s+(?:has|have|is|am|are|was|were)\b"  # More subject-verb agreement
        ]
        grammar_errors = 0
        for pattern in grammar_patterns:
            grammar_errors += len(re.findall(pattern, resume_text, re.IGNORECASE))
            
        # Give up to 15 points for good grammar
        language_score += max(0, 15 - grammar_errors * 5)  # Subtract 5 points per error
        
        # Conciseness score (15%)
        conciseness_score = 0
        
        # Count words
        word_count = len(resume_text.split())
        
        # Ideal range: 300-600 words
        if 300 <= word_count <= 600:
            conciseness_score = 15
        elif word_count < 300:
            conciseness_score = word_count / 300 * 15  # Proportional to minimum
        else:
            conciseness_score = max(0, 15 - (word_count - 600) / 100)  # Subtract 1 point per 100 words over
            
        # Calculate overall score (weighted average)
        overall_score = (content_score * 0.4) + (format_score * 0.3) + (language_score * 0.15) + (conciseness_score * 0.15)
        
        # Prepare result
        scores = {
            "overall": round(overall_score),
            "categories": {
                "content": round(content_score * 100 / 40),  # Convert to 100-point scale
                "format": round(format_score * 100 / 30),
                "language": round(language_score * 100 / 15),
                "conciseness": round(conciseness_score * 100 / 15)
            },
            "summary": get_score_summary(round(overall_score), language)
        }
        
        return scores


def score_resume_with_api(resume_text, language='en'):
    """
    Score CV on a scale from 1-100 using the Groq API.
    
    Args:
        resume_text (str): The CV text to analyze
        language (str): Language for the summary and tips ('en' or 'de')
        
    Returns:
        dict: Score details including overall score and category scores
    """
    try:
        # Prepare a prompt for DeepInfra based on German standards
        system_prompt = """Du bist ein deutscher HR-Experte, der Lebensläufe basierend auf 
        deutschen Bewerbungsstandards bewertet. Bewerte den Lebenslauf in vier Kategorien: 
        Inhalt, Format, Sprache und Prägnanz. Gib Punkte zwischen 0-100."""
        
        user_prompt = f"""Bewerte folgenden Lebenslauf nach deutschen Standards (0-100 Punkte):

{resume_text[:3000]}

Bewerte die folgenden Kategorien:
1. Inhalt (Qualifikationen, Erfahrungen, Leistungen) - 40%
2. Format (Struktur, Übersichtlichkeit, Gliederung) - 30%
3. Sprache (Grammatik, Fachbegriffe, Professionalität) - 15%
4. Prägnanz (Kürze, Relevanz, Fokus) - 15%

Antwort NUR in diesem JSON-Format (ohne weitere Erklärungen):
{{
  "overall": [Gesamtpunktzahl 0-100],
  "categories": {{
    "content": [Punkte 0-100],
    "format": [Punkte 0-100],
    "language": [Punkte 0-100],
    "conciseness": [Punkte 0-100]
  }},
  "summary": [Kurze Zusammenfassung in 1-2 Sätzen]
}}
"""
        
        # Call DeepInfra API
        response = requests.post(
            DEEPINFRA_API_URL,
            headers=deepinfra_headers,
            json={
                "model": DEEPINFRA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            },
            timeout=30
        )
        
        if response.status_code == 200:
            api_response = response.json()
            result_text = api_response["choices"][0]["message"]["content"]
            
            # Extract JSON from response (in case there's additional text)
            json_match = re.search(r'({[\s\S]*})', result_text)
            if json_match:
                result_json = json.loads(json_match.group(1))
                return result_json
            else:
                raise ValueError("Could not extract JSON from API response")
        else:
            # Handle API errors
            logging.error(f"DeepInfra API error ({response.status_code}): {response.text}")
            raise Exception(f"API error: {response.status_code}")
            
    except Exception as e:
        logging.error(f"Error in API-based scoring: {str(e)}")
        raise e


def get_score_summary(score, language='en'):
    """
    Get a summary message based on the overall score.
    
    Args:
        score (int): The overall score (0-100)
        language (str): Language for the summary ('en' or 'de')
        
    Returns:
        str: A summary message
    """
    if language == 'de':
        if score >= 90:
            return "Ausgezeichneter Lebenslauf, der deutsche Standards hervorragend erfüllt. Sofort einsatzbereit."
        elif score >= 80:
            return "Sehr guter Lebenslauf mit wenigen Verbesserungsmöglichkeiten."
        elif score >= 70:
            return "Guter Lebenslauf, der grundlegende Anforderungen erfüllt, aber noch optimiert werden kann."
        elif score >= 60:
            return "Solider Lebenslauf mit mehreren Verbesserungsmöglichkeiten."
        elif score >= 50:
            return "Durchschnittlicher Lebenslauf, der deutliche Überarbeitung benötigt."
        elif score >= 40:
            return "Schwacher Lebenslauf, der erhebliche Verbesserungen erfordert."
        elif score >= 30:
            return "Unzureichender Lebenslauf mit grundlegenden Mängeln."
        else:
            return "Kritisch mangelhafter Lebenslauf, der eine komplette Überarbeitung benötigt."
    else:  # default to English
        if score >= 90:
            return "Excellent CV that meets German standards exceptionally well. Ready for immediate use."
        elif score >= 80:
            return "Very good CV with few areas for improvement."
        elif score >= 70:
            return "Good CV that meets basic requirements but can still be optimized."
        elif score >= 60:
            return "Solid CV with several areas for improvement."
        elif score >= 50:
            return "Average CV that needs significant revision."
        elif score >= 40:
            return "Weak CV that requires substantial improvements."
        elif score >= 30:
            return "Insufficient CV with fundamental deficiencies."
        else:
            return "Critical deficiencies in CV, requires complete revision."


def generate_template_anschreiben(resume_text, job_description, job_title, company_name, skills_info):
    """
    Generate Anschreiben using a template-based approach.
    
    Args:
        resume_text (str): The CV text
        job_description (str): The job description text
        job_title (str): The extracted job title
        company_name (str): The extracted company name
        skills_info (dict): Extracted skills and information
        
    Returns:
        str: Generated Anschreiben text
    """
    # Set up a template structure
    from datetime import datetime
    current_date = datetime.now().strftime('%d.%m.%Y')
    current_date = 'Datum: ' + current_date
    
    # Determine address based on info from resume or use defaults
    sender_name = 'Ihr Name'
    sender_address = 'Ihre Straße 123'
    sender_city = '12345 Ihre Stadt'
    sender_phone = 'Telefon: +49 123 456789'
    sender_email = 'E-Mail: ihre.email@example.com'
    
    # Extract if available in resume
    name_pattern = r'(?:^|\\n)([A-Z][a-z]+ [A-Z][a-z]+)(?:\\n|$)'
    name_match = re.search(name_pattern, resume_text)
    if name_match:
        sender_name = name_match.group(1)
    
    # Build the Anschreiben
    anschreiben = f'''
{sender_name}
{sender_address}
{sender_city}
{sender_phone}
{sender_email}

{current_date}

Bewerbung als {job_title}

Sehr geehrte Damen und Herren,

ich bewerbe mich hiermit um die ausgeschriebene Stelle als {job_title} bei {company_name}, da mein Profil sehr gut zu Ihren Anforderungen passt.
'''
    
    # Add sections based on extracted information
    if skills_info.get('technical_skills'):
        technical_skills = ', '.join(skills_info['technical_skills'][:5])
        anschreiben += f'\nMit meinen fundierten Kenntnissen in {technical_skills} kann ich sofort einen wertvollen Beitrag zu Ihrem Team leisten. '
    
    if skills_info.get('experience') and skills_info['experience']:
        anschreiben += f'Meine Erfahrung als {skills_info["experience"][0]} hat mir die nötigen Kompetenzen vermittelt, um die Herausforderungen dieser Position effektiv zu meistern. '
    
    # Add education if available
    if skills_info.get('education') and skills_info['education']:
        anschreiben += f'Durch mein Studium an der {skills_info["education"][0]} habe ich fundierte theoretische Kenntnisse erworben, die ich in der Praxis erfolgreich anwenden konnte. '
    
    # Languages
    if skills_info.get('languages') and skills_info['languages']:
        languages = ', '.join(skills_info['languages'][:3])
        anschreiben += f'Zusätzlich verfüge ich über {languages} Sprachkenntnisse. '
    
    # Final paragraphs
    anschreiben += f'''
\nGerne überzeuge ich Sie in einem persönlichen Gespräch von meiner Motivation und Eignung für diese Position. Ich freue mich auf Ihre Rückmeldung.

Mit freundlichen Grüßen,

{sender_name}
'''
    
    return anschreiben


def analyze_with_deepinfra(resume_text, language='en'):
    """
    Analyze a CV using Groq's Llama-3-70b model for comprehensive improvement suggestions.
    
    Args:
        resume_text (str): The CV text to analyze
        language (str): Language for the analysis ('en' or 'de')
        
    Returns:
        list: A list of correction suggestions with specific improvements
    """
    logging.info("Analyzing CV with Groq's Llama-3-70b model")
    corrections = []
    
    if groq_client:
        try:
            # Use the native Groq client first (cleaner API)
            # Prepare prompt based on required language
            system_prompt = """You are an expert CV reviewer specializing in German job market standards. 
            Your task is to analyze a CV and provide specific, detailed improvement suggestions.
            Identify issues with content, format, language, achievements, skills presentation, and overall structure.
            For each issue, you must provide the specific text that needs improvement, the suggested fix,
            and a detailed explanation of why the change improves the CV for German employers."""
            
            if language == 'de':
                user_prompt = f"""Analysiere diesen Lebenslauf für den deutschen Arbeitsmarkt:

{resume_text[:2500]}

Gib nur präzise Verbesserungsvorschläge im folgenden JSON-Format zurück:
[
  {{
    "original": "Der zu verbessernde Text",
    "suggestion": "Der verbesserte Text",
    "explanation": "Detaillierte Erklärung, warum diese Änderung den Lebenslauf verbessert",
    "category": "Eine der folgenden Kategorien: grammar, formatting, clarity, professional language, content, achievement, skills"
  }},
  ...weitere Vorschläge
]

Für einen Lebenslauf mit niedrigen Punktzahlen (unter 50/100) solltest du mindestens 7-10 wesentliche Verbesserungsvorschläge machen.
"""
            else:
                user_prompt = f"""Analyze this CV for the German job market:

{resume_text[:2500]}

Return only precise improvement suggestions in the following JSON format:
[
  {{
    "original": "The text that needs improvement",
    "suggestion": "The improved text",
    "explanation": "Detailed explanation why this change improves the CV",
    "category": "One of: grammar, formatting, clarity, professional language, content, achievement, skills"
  }},
  ...more suggestions
]

For a CV with low scores (below 50/100), you should provide at least 7-10 substantial improvement suggestions.
"""

            # Make API call to Groq
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            if response and response.choices and len(response.choices) > 0:
                result_text = response.choices[0].message.content
                
                # Try to extract JSON from the response
                try:
                    # Find JSON array in the response
                    json_match = re.search(r'\[\s*\{.*\}\s*\]', result_text, re.DOTALL)
                    if not json_match:
                        # Try to find in code blocks
                        json_match = re.search(r'```(?:json)?\s*(\[\s*\{.*\}\s*\])```', result_text, re.DOTALL)
                        if json_match:
                            result_text = json_match.group(1)
                        else:
                            raise ValueError("Could not extract JSON from API response")
                    else:
                        result_text = json_match.group(0)
                        
                    # Parse JSON and process each suggestion
                    suggestions = json.loads(result_text)
                    
                    # Process each suggestion
                    corrections = []
                    for suggestion in suggestions:
                        original_text = suggestion.get("original", "")
                        if not original_text:
                            continue
                            
                        # Find position of the original text in the CV
                        start_pos = resume_text.find(original_text)
                        if start_pos == -1:
                            # Try case-insensitive search if exact match fails
                            pattern = re.escape(original_text.lower())
                            matches = list(re.finditer(pattern, resume_text.lower()))
                            if matches:
                                start_pos = matches[0].start()
                            else:
                                # Skip this suggestion if text can't be located
                                logging.warning(f"Original text not found: {original_text[:30]}...")
                                continue
                                
                        # Create correction with detailed information
                        correction = {
                            "original": original_text,
                            "position": {"start": start_pos, "end": start_pos + len(original_text)},
                            "suggestion": suggestion.get("suggestion", ""),
                            "explanation": suggestion.get("explanation", "Improves CV presentation"),
                            "category": suggestion.get("category", "content")
                        }
                        
                        corrections.append(correction)
                    
                    return corrections
                except Exception as e:
                    logging.error(f"Error processing Groq response: {str(e)}")
        
        except Exception as groq_error:
            logging.error(f"Error with Groq native client: {str(groq_error)}")
            logging.error("Falling back to API request method")
    
    # Fall back to API request method if native client fails
    # Prepare the prompt for API based on required language
    system_prompt = """You are an expert CV reviewer specializing in German job market standards. 
    Your task is to analyze a CV and provide specific, detailed improvement suggestions.
    Identify issues with content, format, language, achievements, skills presentation, and overall structure.
    For each issue, you must provide the specific text that needs improvement, the suggested fix,
    and a detailed explanation of why the change improves the CV for German employers."""
    
    if language == 'de':
        user_prompt = f"""Analysiere diesen Lebenslauf für den deutschen Arbeitsmarkt:

{resume_text[:2500]}

Gib nur präzise Verbesserungsvorschläge im folgenden JSON-Format zurück:
[
  {{
    "original": "Der zu verbessernde Text",
    "suggestion": "Der verbesserte Text",
    "explanation": "Detaillierte Erklärung, warum diese Änderung den Lebenslauf verbessert",
    "category": "Eine der folgenden Kategorien: grammar, formatting, clarity, professional language, content, achievement, skills"
  }},
  ...weitere Vorschläge
]

Für einen Lebenslauf mit niedrigen Punktzahlen (unter 50/100) solltest du mindestens 7-10 wesentliche Verbesserungsvorschläge machen.
"""
    else:
        user_prompt = f"""Analyze this CV for the German job market:

{resume_text[:2500]}

Return only precise improvement suggestions in the following JSON format:
[
  {{
    "original": "The text that needs improvement",
    "suggestion": "The improved text",
    "explanation": "Detailed explanation why this change improves the CV",
    "category": "One of: grammar, formatting, clarity, professional language, content, achievement, skills"
  }},
  ...more suggestions
]

For a CV with low scores (below 50/100), you should provide at least 7-10 substantial improvement suggestions.
"""
    
    try:
        # Call DeepInfra API
        response = requests.post(
            DEEPINFRA_API_URL,
            headers=deepinfra_headers,
            json={
                "model": DEEPINFRA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 2000
            },
            timeout=45  # Longer timeout for complex analysis
        )
        
        if response.status_code == 200:
            api_response = response.json()
            result_text = api_response["choices"][0]["message"]["content"]
            
            # Try to extract JSON from the response
            try:
                # Find JSON array in the response
                json_match = re.search(r'\[\s*\{.*\}\s*\]', result_text, re.DOTALL)
                if not json_match:
                    # Try to find in code blocks
                    json_match = re.search(r'```(?:json)?\s*(\[\s*\{.*\}\s*\])```', result_text, re.DOTALL)
                    if json_match:
                        result_text = json_match.group(1)
                    else:
                        raise ValueError("Could not extract JSON from API response")
                else:
                    result_text = json_match.group(0)
                    
                # Parse the extracted JSON
                suggestions = json.loads(result_text)
                
                # Process each suggestion
                for suggestion in suggestions:
                    original_text = suggestion.get("original", "")
                    if not original_text:
                        continue
                        
                    # Find position of the original text in the CV
                    start_pos = resume_text.find(original_text)
                    if start_pos == -1:
                        # Try case-insensitive search if exact match fails
                        pattern = re.escape(original_text.lower())
                        matches = list(re.finditer(pattern, resume_text.lower()))
                        if matches:
                            start_pos = matches[0].start()
                        else:
                            # Skip this suggestion if text can't be located
                            logging.warning(f"Original text not found: {original_text[:30]}...")
                            continue
                            
                    # Create correction with detailed information
                    correction = {
                        "original": original_text,
                        "position": {"start": start_pos, "end": start_pos + len(original_text)},
                        "suggestion": suggestion.get("suggestion", ""),
                        "explanation": suggestion.get("explanation", "Improves CV presentation"),
                        "category": suggestion.get("category", "content")
                    }
                    
                    corrections.append(correction)
                    
                if corrections:
                    return corrections
                else:
                    logging.warning("No valid corrections extracted from DeepInfra response")
                    return []
                    
            except (json.JSONDecodeError, ValueError) as e:
                logging.error(f"Failed to parse DeepInfra response as JSON: {str(e)}")
                logging.debug(f"Raw response: {result_text[:500]}...")
                return []
        else:
            logging.error(f"DeepInfra API error ({response.status_code}): {response.text}")
            return []
            
    except Exception as e:
        logging.error(f"Error in DeepInfra analysis: {str(e)}")
        logging.error(traceback.format_exc())
        return []


def perform_enhanced_analysis(resume_text, language='en'):
    """
    Perform enhanced text analysis for CV issues with a wider range of checks.
    
    Args:
        resume_text (str): The CV text to analyze
        language (str): Language for the analysis ('en' or 'de')
        
    Returns:
        list: List of corrections with detailed improvement suggestions
    """
    corrections = []
    
    # Build a comprehensive set of checks based on German CV standards
    checks = []
    
    # Common weak phrases that should be replaced with strong action verbs
    weak_phrases = [
        # English weak phrases
        {"pattern": r"\bresponsible for\b", "replacement": "managed", 
         "explanation": "Use action verbs instead of passive phrases to show proactive leadership", "category": "clarity"},
        {"pattern": r"\bhelped (with|to)?\b", "replacement": "assisted with", 
         "explanation": "Use more professional terminology to describe your contributions", "category": "professional language"},
        {"pattern": r"\bworked (on|with)\b", "replacement": "developed", 
         "explanation": "Use stronger action verbs to demonstrate your contribution", "category": "clarity"},
        {"pattern": r"\bpart of (a|the) team\b", "replacement": "collaborated with team members to", 
         "explanation": "Specify your role in the team rather than just mentioning team membership", "category": "content"},
        
        # German weak phrases
        {"pattern": r"\bzuständig für\b", "replacement": "verantwortete", 
         "explanation": "Verwenden Sie Aktiv-Formulierungen statt passiver Ausdrücke", "category": "clarity"},
        {"pattern": r"\bhabe (mitge)?arbeitet\b", "replacement": "entwickelte", 
         "explanation": "Nutzen Sie stärkere Verben, um Ihre Beiträge hervorzuheben", "category": "clarity"},
        {"pattern": r"\bwar beteiligt an\b", "replacement": "koordinierte", 
         "explanation": "Verdeutlichen Sie Ihre aktive Rolle statt nur Beteiligung zu erwähnen", "category": "content"},
    ]
    
    # Personal pronouns to avoid in CVs
    pronouns = [
        {"pattern": r"\bi\b", "replacement": "", 
         "explanation": "Avoid first-person pronouns in CVs; start sentences with action verbs instead", "category": "professional language"},
        {"pattern": r"\bmy\b", "replacement": "the", 
         "explanation": "Avoid possessive pronouns in CVs for a more professional tone", "category": "professional language"},
        {"pattern": r"\bich\b", "replacement": "", 
         "explanation": "Vermeiden Sie 'ich' im Lebenslauf; beginnen Sie Sätze direkt mit Verben", "category": "professional language"},
        {"pattern": r"\bmein(e)?\b", "replacement": "die", 
         "explanation": "Vermeiden Sie Possessivpronomen im Lebenslauf", "category": "professional language"},
    ]
    
    # Cliché terms that should be avoided or replaced
    cliches = [
        {"pattern": r"\bteam player\b", "replacement": "collaborative professional", 
         "explanation": "Replace overused clichés with specific examples of collaboration", "category": "content"},
        {"pattern": r"\bthinking outside the box\b", "replacement": "implementing innovative solutions", 
         "explanation": "Avoid clichés and use concrete examples of innovation", "category": "content"},
        {"pattern": r"\bteamfähig\b", "replacement": "arbeitete effektiv im Team bei [Projektname]", 
         "explanation": "Ersetzen Sie Floskeln durch konkrete Beispiele Ihrer Teamarbeit", "category": "content"},
        {"pattern": r"\bhardworking\b", "replacement": "delivered projects consistently ahead of deadline", 
         "explanation": "Show your work ethic through specific achievements rather than generic terms", "category": "content"},
    ]
    
    # Format and capitalization issues
    formatting = [
        {"pattern": r"\bms office\b", "replacement": "Microsoft Office", 
         "explanation": "Use proper capitalization for product names", "category": "formatting"},
        {"pattern": r"\b(java ?script|type ?script)\b", "replacement": "JavaScript", 
         "explanation": "Use correct capitalization for programming languages", "category": "formatting"},
        {"pattern": r"\bc\+\+\b", "replacement": "C++", 
         "explanation": "Use correct capitalization for programming languages", "category": "formatting"},
    ]
    
    # Vague descriptions that need quantification
    vague_terms = [
        {"pattern": r"\b(significantly|substantially|greatly) (improved|increased|decreased|reduced)\b", 
         "replacement": "improved by X%", 
         "explanation": "Quantify your achievements with specific numbers or percentages", "category": "achievement"},
        {"pattern": r"\b(managed|led) a team\b", "replacement": "managed a team of X members", 
         "explanation": "Specify the size of the team you managed for greater impact", "category": "achievement"},
        {"pattern": r"\bverbesserte Prozesse\b", "replacement": "verbesserte Prozesse, was zu einer X% Effizienzsteigerung führte", 
         "explanation": "Quantifizieren Sie Ihre Erfolge mit konkreten Zahlen", "category": "achievement"},
    ]
    
    # German-specific language issues
    german_specific = [
        {"pattern": r"\b(?:gute|sehr gute|ausgezeichnete)\s+kenntnisse\b", 
         "replacement": "Fortgeschrittene Kenntnisse", 
         "explanation": "Verwenden Sie präzisere Begriffe zur Beschreibung Ihrer Fähigkeiten", "category": "professional language"},
        {"pattern": r"\bargts", "replacement": "arbeitet", 
         "explanation": "Verwenden Sie vollständige Wörter statt Abkürzungen", "category": "professional language"},
        {"pattern": r"\binteragirt\b", "replacement": "interagiert", 
         "explanation": "Korrigieren Sie Rechtschreibfehler", "category": "spelling"},
    ]
    
    # Combine all check categories based on language
    if language == 'de':
        checks = weak_phrases + pronouns + cliches + formatting + vague_terms + german_specific
    else:
        checks = weak_phrases + pronouns + cliches + formatting + vague_terms
    
    # Add additional comprehensive checks for low-quality CVs
    # Check for missing sections
    if not re.search(r'\b(?:experience|work|employment|berufserfahrung|arbeitserfahrung|tätigkeiten)\b', resume_text, re.IGNORECASE):
        section_start = resume_text.find("\n\n")
        if section_start == -1:
            section_start = 0
        
        missing_section = {
            "original": resume_text[section_start:section_start+10] + "...",
            "position": {"start": section_start, "end": section_start + 10},
            "suggestion": "BERUFSERFAHRUNG\n[Company Name] | [Position] | [Zeitraum]\n• Verantwortlich für [Hauptaufgabe]\n• Erfolgreich [messbare Leistung] um X% verbessert\n• [Weitere relevante Erfolge]" if language == 'de' else 
                         "PROFESSIONAL EXPERIENCE\n[Company Name] | [Position] | [Time Period]\n• Responsible for [main responsibility]\n• Successfully improved [measurable achievement] by X%\n• [Other relevant achievements]",
            "explanation": "Ein Lebenslauf ohne Berufserfahrung ist unvollständig. Fügen Sie einen klar gekennzeichneten Abschnitt hinzu." if language == 'de' else
                          "A CV without work experience section is incomplete. Add a clearly labeled section.",
            "category": "content"
        }
        corrections.append(missing_section)
    
    if not re.search(r'\b(?:education|ausbildung|bildung|studium|akademisch)\b', resume_text, re.IGNORECASE):
        section_start = resume_text.find("\n\n")
        if section_start == -1:
            section_start = 0
            
        missing_section = {
            "original": resume_text[section_start:section_start+10] + "...",
            "position": {"start": section_start, "end": section_start + 10},
            "suggestion": "BILDUNG\n[Universität/Hochschule] | [Abschluss] | [Zeitraum]\n• Schwerpunkt: [Fachrichtung]\n• Relevante Kurse: [Kursbeispiele]" if language == 'de' else 
                         "EDUCATION\n[University/College] | [Degree] | [Time Period]\n• Focus: [Field of Study]\n• Relevant Coursework: [Course examples]",
            "explanation": "Bildung ist ein wesentlicher Bestandteil eines Lebenslaufs. Fügen Sie einen Abschnitt mit Ihren Abschlüssen hinzu." if language == 'de' else
                          "Education is an essential component of a CV. Add a section with your degrees.",
            "category": "content"
        }
        corrections.append(missing_section)
    
    if not re.search(r'\b(?:skills|fähigkeiten|kenntnisse|kompetenzen)\b', resume_text, re.IGNORECASE):
        section_start = resume_text.find("\n\n")
        if section_start == -1:
            section_start = 0
            
        missing_section = {
            "original": resume_text[section_start:section_start+10] + "...",
            "position": {"start": section_start, "end": section_start + 10},
            "suggestion": "FÄHIGKEITEN\n• Technische Fähigkeiten: [Liste relevanter technischer Fähigkeiten]\n• Sprachen: [Sprachkenntnisse mit Niveauangabe]\n• Soft Skills: [Relevante Soft Skills]" if language == 'de' else 
                         "SKILLS\n• Technical Skills: [List of relevant technical skills]\n• Languages: [Language proficiencies with level]\n• Soft Skills: [Relevant soft skills]",
            "explanation": "Ein Fähigkeiten-Abschnitt hilft Arbeitgebern, Ihre wichtigsten Kompetenzen schnell zu erfassen." if language == 'de' else
                          "A skills section helps employers quickly identify your key competencies.",
            "category": "content"
        }
        corrections.append(missing_section)
    
    # Apply all checks to the resume text
    for check in checks:
        for match in re.finditer(check["pattern"], resume_text, re.IGNORECASE):
            start_pos = match.start()
            end_pos = match.end()
            original_text = resume_text[start_pos:end_pos]
            
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
    
    # Return the corrections, limiting to a reasonable number if there are too many
    max_corrections = 25  # Limit to 25 to avoid overwhelming the user
    if len(corrections) > max_corrections:
        return sorted(corrections, key=lambda x: x["category"])[:max_corrections]
    return corrections
