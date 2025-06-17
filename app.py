from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
import os
import json
import logging
import base64
import io
import math
import tempfile
import requests
import replicate
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

# Configure API keys and tokens
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in environment variables")
if not REPLICATE_API_TOKEN:
    raise ValueError("Missing REPLICATE_API_TOKEN in environment variables")

# Configure AI clients
genai.configure(api_key=GEMINI_API_KEY)
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

app = Flask(__name__)
CORS(app)

def sanitize_prompt(prompt: str) -> str:
    """
    Sanitize architectural prompts to avoid false NSFW content detection.
    Replaces potentially triggering words with safer alternatives while maintaining meaning.
    
    Args:
        prompt: Original architectural prompt
        
    Returns:
        Sanitized prompt with safer terminology
    """
    # Dictionary of problematic words and their safer alternatives
    word_replacements = {
        'penetration': 'transmission',
        'penetrate': 'transmit',
        'penetrating': 'transmitting',
        'penetrates': 'transmits',
        'exposure': 'visibility',
        'exposed': 'visible',
        'intimate': 'close',
        'climax': 'peak',
        'thrust': 'projection',
        'thrusting': 'projecting',
        'erect': 'vertical',
        'erection': 'construction',
        'mounting': 'installation',
        'mounted': 'installed',
        'insertion': 'placement',
        'inserted': 'placed'
    }
    
    sanitized = prompt
    for problematic_word, safe_word in word_replacements.items():
        # Case-insensitive replacement while preserving original case
        import re
        pattern = re.compile(re.escape(problematic_word), re.IGNORECASE)
        
        def replace_func(match):
            original = match.group()
            if original.isupper():
                return safe_word.upper()
            elif original.istitle():
                return safe_word.capitalize()
            else:
                return safe_word
        
        sanitized = pattern.sub(replace_func, sanitized)
    
    return sanitized

def generate_image_from_prompt(
    prompt: str,
    init_image_bytes: Optional[bytes] = None,
    style_strength: float = 0.8
) -> Optional[BytesIO]:
    """Generate an image using Replicate's Flux Canny Pro model.

    Args:
        prompt: Text prompt for image generation
        init_image_bytes: Bytes of the initial image for the control_image
        style_strength: A value from the UI slider (0.1-0.9) to control guidance.

    Returns:
        BytesIO object containing the image data, or None if generation fails
    """
    if not init_image_bytes:
        logger.error("ControlNet generation requires an initial image.")
        return None

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    try:
        temp_file.write(init_image_bytes)
        temp_file.close()

        model_version = "black-forest-labs/flux-canny-pro"
        guidance_value = 5 + (style_strength * 10)

        api_input = {
            "prompt": f"{prompt}, architectural visualization, professional photography, ultra-detailed, 8k, sharp focus",
            "control_image": open(temp_file.name, "rb"),
            "guidance": guidance_value,
            "steps": 30,
            "output_format": "png"
        }

        # Create a prediction and wait for it to complete
        prediction = replicate.predictions.create(
            version=model_version,
            input=api_input
        )
        prediction.wait()

        # Explicitly fetch the final state of the prediction to ensure we have the correct output
        prediction = replicate.predictions.get(prediction.id)

        if prediction.status != 'succeeded':
            logger.error(f"Replicate prediction failed: {prediction.error}")
            return None

        # The output is a single URL string, not a list
        output_url = prediction.output
        response = requests.get(output_url)
        response.raise_for_status()

        return BytesIO(response.content)

    except Exception as e:
        logger.error(f"Error generating image with Replicate API: {str(e)}", exc_info=True)
        return None
    finally:
        try:
            os.unlink(temp_file.name)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file: {e}")


def clean_json_response(response_text: str) -> str:
    """Clean the response text by removing Markdown code fences and extra whitespace.
    
    Args:
        response_text: Raw text response that might contain Markdown formatting
        
    Returns:
        Cleaned JSON string ready for parsing
    """
    # Remove leading/trailing whitespace
    cleaned = response_text.strip()
    
    # Handle ```json ... ``` pattern
    if cleaned.startswith('```json') and cleaned.endswith('```'):
        # Remove the first ```json and last ```, then strip any extra whitespace
        cleaned = cleaned[7:-3].strip()
    # Handle ``` ... ``` pattern (without json)
    elif cleaned.startswith('```') and cleaned.endswith('```'):
        cleaned = cleaned[3:-3].strip()
    
    return cleaned


def get_ai_design_logic(
    facade_length: float,
    floor_height: float,
    num_floors: int,
    building_program: str,
    climate_type: str,
    window_ratio: int,
    primary_rhythm: str,
    primary_materials: str,
    image_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate design logic using Gemini AI based on the given constraints.
    
    Args:
        facade_length: Length of the facade in meters
        floor_height: Height of each floor in meters
        num_floors: Number of floors
        building_program: Type of building (e.g., 'Office', 'Residential')
        climate_type: Climate type (e.g., 'Temperate', 'Tropical')
        window_ratio: Target window-to-wall ratio (0-100)
        primary_rhythm: Primary facade rhythm (e.g., 'Regular', 'Irregular')
        primary_materials: Primary construction materials
        image_path: Optional path to uploaded massing image
        
    Returns:
        Dictionary containing the AI's design logic
    """
    # Create a detailed prompt for the AI
    prompt = f"""You are an expert architectural designer. Based on the following parameters:
    
    Building Program: {building_program}
    Climate Type: {climate_type}
    Facade Length: {facade_length} meters
    Floor Height: {floor_height} meters
    Number of Floors: {num_floors}
    Target Window-to-Wall Ratio: {window_ratio}%
    Primary Rhythm: {primary_rhythm}
    Primary Materials: {primary_materials}
    
    Please analyze these parameters and provide design recommendations in the following JSON format:
    {{
        "wwr_actual_percent": <number between 0-100>,
        "daylight_score_estimate": <number between 0-100>,
        "solar_gain_qualitative": <"Low" | "Medium" | "High">,
        "generated_style_description": "<brief description of the recommended architectural style>"
    }}
    
    Consider the following in your analysis:
    - The building program's functional requirements
    - Climate-appropriate design strategies
    - The relationship between window placement and energy efficiency
    - How the primary materials influence the design
    - The desired rhythm and its impact on the facade composition
    - Any relevant building codes or standards
    
    Return ONLY the JSON object with no additional text or markdown formatting.
    """
    
    try:
        # Initialize the model
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        
        # If there's an image, include it in the prompt
        if image_path and os.path.exists(image_path):
            # For now, we'll just mention the image in the prompt
            # In a future update, we can actually process the image
            prompt += "\n\nNote: A massing image was also provided for reference."
        
        # Sanitize the prompt
        sanitized_prompt = sanitize_prompt(prompt)
        
        # Generate the response
        response = model.generate_content(sanitized_prompt)
        
        if not hasattr(response, 'text') or not response.text:
            raise ValueError("No response text received from the AI model")
        
        # Parse the JSON response
        try:
            # Clean the response text before parsing
            cleaned_response = clean_json_response(response.text)
            design_data = json.loads(cleaned_response)
            
            # Validate the response structure
            required_keys = [
                'wwr_actual_percent', 
                'daylight_score_estimate', 
                'solar_gain_qualitative', 
                'generated_style_description'
            ]
            
            if not all(key in design_data for key in required_keys):
                raise ValueError("Missing required fields in AI response")
                
            # Validate data types
            if not (isinstance(design_data['wwr_actual_percent'], (int, float)) and 
                   0 <= design_data['wwr_actual_percent'] <= 100):
                raise ValueError("Invalid value for wwr_actual_percent")
                
            if not (isinstance(design_data['daylight_score_estimate'], (int, float)) and 
                   0 <= design_data['daylight_score_estimate'] <= 100):
                raise ValueError("Invalid value for daylight_score_estimate")
                
            if design_data['solar_gain_qualitative'].lower() not in ['low', 'medium', 'high']:
                raise ValueError("solar_gain_qualitative must be 'Low', 'Medium', or 'High'")
            
            return design_data
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse AI response as JSON: {response.text}")
            raise ValueError(f"Invalid JSON response from AI: {str(e)}")
            
    except Exception as e:
        logging.error(f"Error in get_ai_design_logic: {str(e)}")
        raise

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate a 2D facade pattern based on form data and optional image upload.
    Expected multipart/form-data with the following fields:
    - facade_length: float
    - floor_height: float
    - num_floors: int
    - building_program: str
    - climate_type: str
    - window_ratio: int
    - primary_rhythm: str
    - primary_materials: str
    - massing_image: file (optional)
    
    Returns: JSON response with received data and filename if image was uploaded
    """
    # Create uploads directory if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Initialize variables
    filename = None
    
    try:
        # Get form data
        facade_length = float(request.form.get('facadeLength', 20))
        floor_height = float(request.form.get('floorHeight', 3.5))
        num_floors = int(request.form.get('numFloors', 5))
        building_program = request.form.get('buildingProgram', 'office')
        climate_type = request.form.get('climateType', 'temperate')
        window_ratio = int(request.form.get('windowRatio', 40))
        primary_rhythm = request.form.get('primaryRhythm', 'vertical')
        primary_materials = request.form.get('primaryMaterials', 'Glass and Aluminum')
        # Validate required fields
        required_fields = {
            'facade_length': facade_length,
            'floor_height': floor_height,
            'num_floors': num_floors,
            'building_program': building_program,
            'climate_type': climate_type,
            'window_ratio': window_ratio,
            'primary_rhythm': primary_rhythm,
            'primary_materials': primary_materials
        }
        
        missing_fields = [field for field, value in required_fields.items() if value is None]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields
            }), 400
            
        # Validate numerical values
        if facade_length <= 0 or floor_height <= 0 or num_floors <= 0:
            return jsonify({
                'error': 'Facade length, floor height, and number of floors must be positive numbers'
            }), 400
            
        if window_ratio < 0 or window_ratio > 100:
            return jsonify({
                'error': 'Window to wall ratio must be between 0 and 100'
            }), 400
        
        # Handle file upload if present
        if 'massingImage' in request.files:
            file = request.files['massingImage']
            if file.filename != '':
                if file and allowed_file(file.filename):
                    # Save the file
                    filename = os.path.join(UPLOAD_FOLDER, file.filename)
                    file.save(filename)
                else:
                    return jsonify({
                        'error': 'Invalid file type. Allowed types are: ' + ', '.join(ALLOWED_EXTENSIONS)
                    }), 400
            else:
                return jsonify({
                    'error': 'Massing image is required for facade generation'
                }), 400
        else:
            return jsonify({
                'error': 'Massing image is required for facade generation'
            }), 400
        
        # Get AI design logic
        design_data = get_ai_design_logic(
            facade_length=facade_length,
            floor_height=floor_height,
            num_floors=num_floors,
            building_program=building_program,
            climate_type=climate_type,
            window_ratio=window_ratio,
            primary_rhythm=primary_rhythm,
            primary_materials=primary_materials,
            image_path=filename
        )
        
        # Calculate overall performance score
        solar_gain_mapping = {
            'Low': 25,
            'Medium': 50, 
            'High': 75
        }
        solar_gain_numeric = solar_gain_mapping.get(design_data['solar_gain_qualitative'], 50)
        
        daylight_score = design_data['daylight_score_estimate']
        overall_performance_score = round((daylight_score * 0.6) + ((100 - solar_gain_numeric) * 0.4), 1)
        
        # 1. Prepare dashboard data with calculations
        dashboard_data = {
            # Input parameters
            'facade_length': facade_length,
            'floor_height': floor_height,
            'num_floors': num_floors,
            'building_program': building_program,
            'climate_type': climate_type,
            'target_wwr': window_ratio,
            'primary_rhythm': primary_rhythm,
            'primary_materials': primary_materials,
            
            # AI-generated values
            'wwr_actual_percent': design_data['wwr_actual_percent'],
            'daylight_score_estimate': design_data['daylight_score_estimate'],
            'solar_gain_qualitative': design_data['solar_gain_qualitative'],
            'generated_style_description': design_data['generated_style_description'],
            
            # Calculated metrics
            'total_facade_area': facade_length * floor_height * num_floors,
            'glass_area': None,  # Will be calculated
            'wall_area': None,   # Will be calculated
            'actual_wwr': None,  # Will match wwr_actual_percent but in decimal
            'overall_performance_score': overall_performance_score,
            'solar_gain_numeric': solar_gain_numeric
        }
        
        # Calculate glass and wall areas based on WWR
        total_area = dashboard_data['total_facade_area']
        wwr_decimal = dashboard_data['wwr_actual_percent'] / 100.0
        
        dashboard_data['glass_area'] = round(total_area * wwr_decimal, 2)
        dashboard_data['wall_area'] = round(total_area - dashboard_data['glass_area'], 2)
        dashboard_data['actual_wwr'] = wwr_decimal
        
        # Generate image if we have a style description
        base64_image = None
        if 'generated_style_description' in design_data:
            try:
                # Read the uploaded image if it exists
                init_image_bytes = None
                if filename and os.path.exists(filename):
                    with open(filename, 'rb') as f:
                        init_image_bytes = f.read()
                
                # Create a new final_image_prompt variable that combines the Gemini AI description with the user's primary materials input
                final_image_prompt = f"A photorealistic architectural visualization of a complete building facade with {design_data['generated_style_description']}. The primary materials are explicitly {primary_materials}. The input image shows the full building massing and overall form - apply the facade design to the entire visible building surface, not just a portion. This is a complete building elevation view, architectural rendering, professional photography, ultra-detailed, 8k, sharp focus."
                
                # Sanitize the prompt
                sanitized_prompt = sanitize_prompt(final_image_prompt)
                
                # Generate image using Flux Canny Pro with the uploaded image
                img_data = generate_image_from_prompt(
                    prompt=sanitized_prompt,
                    init_image_bytes=init_image_bytes
                )
                
                if img_data:
                    img_data.seek(0)  # Ensure we're at the start of the BytesIO
                    base64_image = base64.b64encode(img_data.read()).decode('utf-8')
                    img_data.close()  # Clean up the BytesIO object
                    
            except Exception as e:
                logging.error(f"Error generating image with Stability AI: {str(e)}")
                # Continue without image if generation fails
        
        # 3. Prepare final response
        response_data = {
            'status': 'success',
            'dashboard_data': dashboard_data,
            'base64_image': base64_image,
            'image_generated': base64_image is not None
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        # Log the error for debugging
        error_details = {
            "error": "An error occurred while processing your request",
            "type": type(e).__name__,
            "message": str(e)
        }
        
        # Add more context if available
        if 'response' in locals() and response is not None and hasattr(response, 'text') and response.text:
            error_details["response_text"] = response.text[:500]  # Include first 500 chars
            
        logging.error(f"Error in generate: {error_details}")
        
        # Clean up uploaded file if there was an error
        if 'filename' in locals() and filename and os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception as cleanup_error:
                logging.error(f"Error cleaning up file {filename}: {str(cleanup_error)}")
        
        # Return error response
        return jsonify(error_details), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
