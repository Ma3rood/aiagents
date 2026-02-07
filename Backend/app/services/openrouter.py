import httpx
import json
import re
import time
from typing import Dict, Any, Optional, List
from app.core.config import settings
from app.core.logging_config import get_logger
from app.core.analytics import get_analytics_service

logger = get_logger(__name__)


def _fix_trailing_comma_json(s: str) -> str:
    """Fix common LLM JSON issues: trailing commas, missing commas between array elements."""
    s = re.sub(r",\s*]", "]", s)
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r"}\s*(?:\r?\n\s*)+{", "},\n{", s)
    s = re.sub(r"}\s+{", "},{", s)
    return s


class OpenRouterService:
    """Service for interacting with OpenRouter API using Qwen3 VL model"""
    
    MODEL = "qwen/qwen3-vl-235b-a22b-instruct"  # Vision model for image processing
    TEXT_MODEL = "docker.io/ai/qwen3-vl:latest"  # Text model for translation and text tasks
    EMBEDDING_MODEL = "docker.io/ai/qwen3-embedding:8B-Q4_K_M"  # Embedding model for semantic search
    
    # Root categories for marketplace classification
    ROOT_CATEGORIES = [
        "Antiques & collectables",
        "Art",
        "Baby-gear",
        "Books",
        "Building-renovation",
        "Business-farming-industry",
        "Clothing & Fashion",
        "Computers",
        "Crafts",
        "Electronics-photography",
        "Flatmates-wanted",
        "Gaming",
        "Health-beauty",
        "Home & Living",
        "Jewellery & watches",
        "Mobile phones",
        "Movies & TV",
        "Music-instruments",
        "Pets & animals",
        "Pottery & Glass",
        "Sports",
        "Toys & models",
        "Travel-events-activities"
    ]
    
    def __init__(self):
        self.api_key = settings.OPENROUTER_API_KEY
        if not self.api_key:
            logger.error("OPENROUTER_API_KEY is not set in environment variables")
            raise ValueError("OPENROUTER_API_KEY is not set in environment variables")
        self.base_url = settings.OPENROUTER_BASE_URL
        self.embedding_url = settings.OPENROUTER_EMBEDDING_URL
        logger.debug(f"OpenRouterService initialized - Base URL: {self.base_url}")
    
    async def extract_form_fields_from_images(
        self, 
        image_urls: List[str], 
        language: str
    ) -> Dict[str, Any]:
        """
        Extract marketplace listing form fields from multiple images of the same product using Qwen3 VL model.
        
        Args:
            image_urls: List of URLs of images to analyze (all images should be of the same product)
            language: Selected language for the response
            
        Returns:
            Dictionary containing extracted form field values
        """
        start_time = time.time()
        logger.info(
            f"Extracting form fields from images - Image count: {len(image_urls)}, "
            f"Language: {language}, Model: {self.MODEL}"
        )
        
        # Construct the prompt for marketplace listing extraction
        prompt = self._build_marketplace_prompt(language, len(image_urls))
        input_char_count = len(prompt)
        logger.debug(f"Built marketplace prompt (length: {input_char_count} chars)")
        
        analytics_service = get_analytics_service()
        status = "success"
        error_message = None
        output_char_count = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cached_tokens = 0
        reasoning_tokens = 0
        cost = 0.0
        
        # Prepare the content array with text prompt and all images
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        
        # Add all image URLs to the content
        for image_url in image_urls:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            })
        
        # Prepare the messages for the API
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",  # Optional: for OpenRouter analytics
            "X-Title": "Ma3rood AI Agents Marketplace Agent"  # Optional: for OpenRouter analytics
        }
        
        payload = {
            "model": self.MODEL,
            "messages": messages,
            "temperature": 0.3,  # Lower temperature for more consistent extraction
            "max_tokens": 2000,
            "usage": {
                "include": True  # Enable usage accounting as per OpenRouter docs
            }
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                logger.debug(f"Sending request to OpenRouter API - Model: {self.MODEL}")
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                logger.debug("Received response from OpenRouter API")
                
                # Extract token usage for analytics (enabled via usage.include)
                if "usage" in result:
                    usage = result["usage"]
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    cost = usage.get("cost", 0)
                    
                    # Extract detailed token information
                    prompt_details = usage.get("prompt_tokens_details", {})
                    completion_details = usage.get("completion_tokens_details", {})
                    
                    cached_tokens = prompt_details.get("cached_tokens", 0)
                    reasoning_tokens = completion_details.get("reasoning_tokens", 0)
                else:
                    logger.warning("Token usage information not available in API response")
                
                # Extract the content from the response
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    logger.debug(f"Received content from API (length: {len(content)} chars)")
                    
                    # Try to parse JSON from the response
                    try:
                        # The model might return JSON directly or wrapped in markdown
                        content = content.strip()
                        if content.startswith("```json"):
                            content = content[7:]  # Remove ```json
                        if content.startswith("```"):
                            content = content[3:]  # Remove ```
                        if content.endswith("```"):
                            content = content[:-3]  # Remove closing ```
                        content = content.strip()
                        
                        form_data = json.loads(content)
                        output_char_count = len(json.dumps(form_data, ensure_ascii=False))
                        logger.info(f"Successfully parsed form data - Keys: {list(form_data.keys()) if isinstance(form_data, dict) else 'N/A'}")
                        
                        # Log analytics
                        time_taken_seconds = time.time() - start_time
                        analytics_service.log_analytics(
                            agent_type="image_to_form",
                            model=self.MODEL,
                            provider="OpenRouter",
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            cached_tokens=cached_tokens,
                            reasoning_tokens=reasoning_tokens,
                            cost=cost,
                            input_char_count=input_char_count,
                            output_char_count=output_char_count,
                            time_taken_seconds=time_taken_seconds,
                            target_language=language,
                            status=status,
                            error_message=error_message
                        )
                        
                        return form_data
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON from API response: {str(e)}")
                        logger.debug(f"Raw content (first 500 chars): {content[:500]}")
                        output_char_count = len(content)
                        # If not JSON, return as structured text
                        result_data = {
                            "raw_response": content,
                            "parsed": False
                        }
                        
                        # Log analytics even for partial success
                        time_taken_seconds = time.time() - start_time
                        analytics_service.log_analytics(
                            agent_type="image_to_form",
                            model=self.MODEL,
                            provider="OpenRouter",
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            cached_tokens=cached_tokens,
                            reasoning_tokens=reasoning_tokens,
                            cost=cost,
                            input_char_count=input_char_count,
                            output_char_count=output_char_count,
                            time_taken_seconds=time_taken_seconds,
                            target_language=language,
                            status="partial_success",
                            error_message=f"JSON parse error: {str(e)}"
                        )
                        
                        return result_data
                else:
                    logger.error("No choices in API response")
                    status = "error"
                    error_message = "No choices in API response"
                    raise ValueError("No choices in API response")
                    
            except httpx.HTTPStatusError as e:
                error_detail = f"OpenRouter API error: {e.response.status_code}"
                logger.error(
                    f"HTTP error from OpenRouter API - Status: {e.response.status_code}, "
                    f"URL: {self.base_url}"
                )
                if e.response.text:
                    try:
                        error_data = e.response.json()
                        error_detail = error_data.get("error", {}).get("message", error_detail)
                        logger.error(f"API error details: {error_detail}")
                    except:
                        error_detail = f"{error_detail} - {e.response.text}"
                        logger.error(f"API error response text: {e.response.text}")
                
                # Log analytics for error
                time_taken_seconds = time.time() - start_time
                analytics_service.log_analytics(
                    agent_type="image_to_form",
                    model=self.MODEL,
                    provider="OpenRouter",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cached_tokens=cached_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cost=cost,
                    input_char_count=input_char_count,
                    output_char_count=output_char_count,
                    time_taken_seconds=time_taken_seconds,
                    target_language=language,
                    status="error",
                    error_message=error_detail
                )
                
                raise Exception(error_detail)
            except httpx.RequestError as e:
                error_detail = f"Request error: {str(e)}"
                logger.error(f"Request error to OpenRouter API: {str(e)}", exc_info=True)
                
                # Log analytics for error
                time_taken_seconds = time.time() - start_time
                analytics_service.log_analytics(
                    agent_type="image_to_form",
                    model=self.MODEL,
                    provider="OpenRouter",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cached_tokens=cached_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cost=cost,
                    input_char_count=input_char_count,
                    output_char_count=output_char_count,
                    time_taken_seconds=time_taken_seconds,
                    target_language=language,
                    status="error",
                    error_message=error_detail
                )
                
                raise Exception(error_detail)
            except Exception as e:
                # Log analytics for any other error
                time_taken_seconds = time.time() - start_time
                analytics_service.log_analytics(
                    agent_type="image_to_form",
                    model=self.MODEL,
                    provider="OpenRouter",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cached_tokens=cached_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cost=cost,
                    input_char_count=input_char_count,
                    output_char_count=output_char_count,
                    time_taken_seconds=time_taken_seconds,
                    target_language=language,
                    status="error",
                    error_message=str(e)
                )
                raise
    
    def _build_marketplace_prompt(self, language: str, image_count: int = 1) -> str:
        """Build the prompt for extracting marketplace listing form fields"""
        
        image_text = "these images" if image_count > 1 else "this image"
        image_instruction = f"Analyze all {image_count} images provided" if image_count > 1 else "Analyze this image"
        
        prompt = f"""{image_instruction} and extract all relevant information for a marketplace listing form. 
All {image_text} show the same product from different angles or views. Combine information from all images to get a complete picture of the product.
Respond ONLY with a valid JSON object in {language} language. Do not include any markdown formatting or explanations outside the JSON.

REQUIRED FIELDS (always include these):
- title: Product/item title (string, required)
- description: Detailed, marketing-friendly description of the product/item (string, required). Make it compelling and appealing to buyers, highlight key benefits/features clearly visible in the images.
- price: Price if visible (number, or null if not found)
- category: Product category (string, e.g., "Electronics", "Clothing", "Furniture", "Vehicles", etc.)
- condition: Condition of the item. Must be one of the following exact values (string, or null if not determinable):
  * "Brand New or Unused": never opened or used
  * "Like New": opened but looks and works like new
  * "Gently Used or Excellent Condition": minor signs of use
  * "Good Condition": visible wear but fully functional
  * "Fair Condition": heavily used but still works
  * "For Parts or Not Working": damaged or needs repair
  * "Not Applicable": condition does not apply to this item
- location: Location if visible (string or null)
- tags: Array of relevant tags/keywords (array of strings)
- quantity: Quantity available if mentioned (number or null)

ATTRIBUTES (include ONLY relevant attributes):
Analyze the product in the image and extract ONLY the attributes that are relevant to this specific product. 
Attributes can be from the following list OR any other relevant attributes not listed here:

Possible attributes (use only if relevant):
- Brand
- Model
- Color
- Size
- Weight
- Material
- Memory/RAM
- Storage
- Hard Drive Size
- Processor
- Cores
- Screen Size
- Resolution
- Battery Capacity
- Operating System
- Network Type
- Camera
- Battery Life
- Gender
- Season
- Year
- Mileage
- Fuel Type
- Transmission
- Dimensions
- Condition Details
- Warranty
- Accessories Included

CRITICAL INSTRUCTIONS:
1. Analyze ALL provided images together to extract comprehensive information about the product
2. For the "condition" field, you MUST use one of the exact values listed above. Analyze the images carefully to determine the most appropriate condition based on visible wear, packaging, and usage signs
3. Include ONLY attributes that are relevant to the product shown in the images
4. Do NOT include attributes that are not applicable (e.g., don't include "Mileage" for a smartphone)
5. You can include attributes NOT in the list above if they are relevant to the product
6. Attribute names should be in English (camelCase or Title Case)
7. Attribute values should be in {language} language (except for condition which must use the exact English values listed)
8. If an attribute is not visible or cannot be determined from any of the images, do NOT include it in the response
9. All text should be extracted in {language} language (except condition field)
10. Be accurate and only extract information that is clearly visible in the images
11. Use information from all images to build a complete product description

Response format:
{{
  "title": "Product Title",
  "description": "Detailed description...",
  "price": 99.99,
  "category": "Electronics",
  "condition": "Brand New or Unused",
  "location": "City, Country",
  "tags": ["tag1", "tag2"],
  "quantity": 1,
  "attributes": {{
    "Brand": "Brand Name",
    "Model": "Model XYZ",
    "Color": "Black",
    "Storage": "256GB",
    "Screen Size": "6.1 inches",
    "Battery Capacity": "4000 mAh"
    // Include ONLY relevant attributes for this product
  }}
}}

Example for a smartphone:
{{
  "title": "iPhone 15 Pro",
  "description": "Latest iPhone with advanced features...",
  "price": 999.99,
  "category": "Electronics",
  "condition": "Brand New or Unused",
  "location": "New York, USA",
  "tags": ["smartphone", "apple", "iphone"],
  "quantity": 1,
  "attributes": {{
    "Brand": "Apple",
    "Model": "iPhone 15 Pro",
    "Color": "Titanium Blue",
    "Storage": "256GB",
    "Screen Size": "6.1 inches",
    "Battery Capacity": "3274 mAh",
    "Operating System": "iOS 17"
  }}
}}

Example for clothing:
{{
  "title": "Nike Running Shoes",
  "description": "Comfortable running shoes...",
  "price": 129.99,
  "category": "Clothing",
  "condition": "Brand New or Unused",
  "location": "Los Angeles, USA",
  "tags": ["shoes", "running", "nike"],
  "quantity": 1,
  "attributes": {{
    "Brand": "Nike",
    "Color": "Black/White",
    "Size": "10",
    "Gender": "Unisex",
    "Material": "Mesh and Synthetic"
  }}
}}"""

        return prompt

    async def translate_listing_fields(
        self,
        fields: Dict[str, str],
        listing_details: Optional[Dict[str, Any]],
        target_language: str,
        listing_type: str = "Marketplace",
        model: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Translate listing fields to the target language while preserving context and terminology.
        
        Args:
            fields: Dictionary of field_name -> text to translate
            listing_details: Optional context about the listing (category, product type, etc.)
            target_language: Target language for translation
            listing_type: Type of listing - Marketplace, Motors, Services, Jobs, or Property
            model: Optional model name to use (defaults to TEXT_MODEL)
            
        Returns:
            Dictionary of field_name -> translated text
        """
        start_time = time.time()
        model_to_use = model or self.TEXT_MODEL
        logger.info(
            f"Translating listing fields - Target language: {target_language}, "
            f"Listing type: {listing_type}, "
            f"Fields count: {len(fields)}, Model: {model_to_use}"
        )
        
        prompt = self._build_translation_prompt(fields, listing_details, target_language, listing_type)
        input_char_count = len(prompt) + len(json.dumps(fields, ensure_ascii=False))
        if listing_details:
            input_char_count += len(json.dumps(listing_details, ensure_ascii=False))
        logger.debug(f"Built translation prompt (length: {len(prompt)} chars)")
        
        analytics_service = get_analytics_service()
        status = "success"
        error_message = None
        output_char_count = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cached_tokens = 0
        reasoning_tokens = 0
        cost = 0.0
        
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "Ma3rood AI Agents Translation Agent"
        }
        
        # Use provided model or default to TEXT_MODEL
        model_to_use = model or self.TEXT_MODEL
        
        payload = {
            "model": model_to_use,
            "messages": messages,
            "temperature": 0.2,  # Lower temperature for accurate translations
            "max_tokens": 4000,
            "usage": {
                "include": True
            }
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                logger.debug(f"Sending translation request to OpenRouter API - Model: {model_to_use}")
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                logger.debug("Received translation response from OpenRouter API")
                
                # Extract token usage for analytics
                if "usage" in result:
                    usage = result["usage"]
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    cost = usage.get("cost", 0)
                    
                    # Extract detailed token information
                    prompt_details = usage.get("prompt_tokens_details", {})
                    completion_details = usage.get("completion_tokens_details", {})
                    
                    cached_tokens = prompt_details.get("cached_tokens", 0)
                    reasoning_tokens = completion_details.get("reasoning_tokens", 0)
                
                # Extract the content from the response
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    logger.debug(f"Received translation content (length: {len(content)} chars)")
                    
                    # Parse JSON from response
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                    
                    translated_data = json.loads(content)
                    translations = translated_data.get("translations", {})
                    output_char_count = len(json.dumps(translations, ensure_ascii=False))
                    logger.info(f"Translation completed - Translated fields: {len(translations)}")
                    
                    # Log analytics
                    time_taken_seconds = time.time() - start_time
                    analytics_service.log_analytics(
                        agent_type="translation",
                        model=model_to_use,
                        provider="OpenRouter",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        cached_tokens=cached_tokens,
                        reasoning_tokens=reasoning_tokens,
                        cost=cost,
                        input_char_count=input_char_count,
                        output_char_count=output_char_count,
                        time_taken_seconds=time_taken_seconds,
                        target_language=target_language,
                        listing_type=listing_type,
                        fields_count=len(fields),
                        status=status,
                        error_message=error_message
                    )
                    
                    return translations
                else:
                    logger.error("No choices in translation API response")
                    status = "error"
                    error_message = "No choices in API response"
                    raise ValueError("No choices in API response")
                    
            except httpx.HTTPStatusError as e:
                error_detail = f"OpenRouter API error: {e.response.status_code}"
                logger.error(
                    f"HTTP error from OpenRouter API (translation) - Status: {e.response.status_code}, "
                    f"URL: {self.base_url}"
                )
                if e.response.text:
                    try:
                        error_data = e.response.json()
                        error_detail = error_data.get("error", {}).get("message", error_detail)
                        logger.error(f"API error details: {error_detail}")
                    except:
                        error_detail = f"{error_detail} - {e.response.text}"
                        logger.error(f"API error response text: {e.response.text}")
                
                # Log analytics for error
                time_taken_seconds = time.time() - start_time
                analytics_service.log_analytics(
                    agent_type="translation",
                    model=model_to_use,
                    provider="OpenRouter",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cached_tokens=cached_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cost=cost,
                    input_char_count=input_char_count,
                    output_char_count=output_char_count,
                    time_taken_seconds=time_taken_seconds,
                    target_language=target_language,
                    listing_type=listing_type,
                    fields_count=len(fields),
                    status="error",
                    error_message=error_detail
                )
                
                raise Exception(error_detail)
            except httpx.RequestError as e:
                error_detail = f"Request error: {str(e)}"
                logger.error(f"Request error to OpenRouter API (translation): {str(e)}", exc_info=True)
                
                # Log analytics for error
                time_taken_seconds = time.time() - start_time
                analytics_service.log_analytics(
                    agent_type="translation",
                    model=model_to_use,
                    provider="OpenRouter",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cached_tokens=cached_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cost=cost,
                    input_char_count=input_char_count,
                    output_char_count=output_char_count,
                    time_taken_seconds=time_taken_seconds,
                    target_language=target_language,
                    listing_type=listing_type,
                    fields_count=len(fields),
                    status="error",
                    error_message=error_detail
                )
                
                raise Exception(error_detail)
            except json.JSONDecodeError as e:
                error_detail = f"Failed to parse translation response: {str(e)}"
                logger.error(f"Failed to parse JSON from translation response: {str(e)}")
                logger.debug(f"Raw content (first 500 chars): {content[:500] if 'content' in locals() else 'N/A'}")
                
                # Log analytics for error
                time_taken_seconds = time.time() - start_time
                if 'content' in locals():
                    output_char_count = len(content)
                analytics_service.log_analytics(
                    agent_type="translation",
                    model=model_to_use,
                    provider="OpenRouter",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cached_tokens=cached_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cost=cost,
                    input_char_count=input_char_count,
                    output_char_count=output_char_count,
                    time_taken_seconds=time_taken_seconds,
                    target_language=target_language,
                    listing_type=listing_type,
                    fields_count=len(fields),
                    status="error",
                    error_message=error_detail
                )
                
                raise Exception(error_detail)
            except Exception as e:
                # Log analytics for any other error
                time_taken_seconds = time.time() - start_time
                analytics_service.log_analytics(
                    agent_type="translation",
                    model=model_to_use,
                    provider="OpenRouter",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cached_tokens=cached_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cost=cost,
                    input_char_count=input_char_count,
                    output_char_count=output_char_count,
                    time_taken_seconds=time_taken_seconds,
                    target_language=target_language,
                    listing_type=listing_type,
                    fields_count=len(fields),
                    status="error",
                    error_message=str(e)
                )
                raise
    
    def _get_listing_type_instructions(self, listing_type: str) -> List[str]:
        """Get listing type specific translation instructions as a list of guideline points"""
        
        instructions_map = {
            "Marketplace": [
                "Focus on product descriptions, features, and specifications",
                "Keep pricing and currency information clear",
                "Preserve condition descriptions accurately",
                "Maintain product categories and tags appropriately"
            ],
            
            "Motors": [
                "Focus on vehicle specifications, features, and condition",
                "Keep pricing and currency information clear",
                "Preserve vehicle condition, accident history, and service records accurately",
                "Maintain vehicle categories (car, motorcycle, truck, etc.) and features appropriately"
            ],
            
            "Services": [
                "Focus on service descriptions, qualifications, and offerings",
                "Preserve professional certifications, licenses, and credentials",
                "Maintain accuracy for service areas, availability, and pricing",
                "Keep service categories and specializations clear",
                "Preserve business hours, contact information, and service terms accurately"
            ],
            
            "Jobs": [
                "Focus on job descriptions, requirements, and benefits",
                "Preserve job titles, company names, and location information",
                "Maintain accuracy for salary ranges, work hours, and employment type",
                "Keep job categories, industries, and skill requirements clear",
                "Preserve qualifications, experience requirements, and application details accurately"
            ],
            
            "Property": [
                "Focus on property descriptions, features, and specifications",
                "Preserve property addresses, property IDs, and reference numbers",
                "Keep pricing, currency, and payment terms clear",
                "Preserve property condition, age, and maintenance history accurately",
                "Maintain property types (apartment, house, commercial, etc.) and features appropriately",
                "Preserve location details, neighborhood information, and nearby amenities"
            ]
        }
        
        return instructions_map.get(listing_type, instructions_map["Marketplace"])
    
    def _build_translation_prompt(
        self, 
        fields: Dict[str, str], 
        listing_details: Optional[Dict[str, Any]], 
        target_language: str,
        listing_type: str = "Marketplace"
    ) -> str:
        """Build the prompt for context-aware translation of listing fields based on listing type"""
        
        # Build context section from listing details - include all fields dynamically
        # Only add context if listing_details is not None and has at least one field
        context_section = ""
        if listing_details and len(listing_details) > 0:
            context_parts = []
            for key, value in listing_details.items():
                # Skip None values and empty strings
                if value is not None and value != "":
                    # Format the value appropriately
                    if isinstance(value, (dict, list)):
                        # For nested structures, use JSON formatting
                        formatted_value = json.dumps(value, ensure_ascii=False, indent=2)
                        context_parts.append(f"- {key}:\n{formatted_value}")
                    else:
                        context_parts.append(f"- {key}: {value}")
            
            # Only add context section if we have at least one valid field
            if context_parts:
                context_section = f"""
LISTING CONTEXT (use this to understand the product and maintain appropriate terminology):
{chr(10).join(context_parts)}
"""
        
        # Build fields section
        fields_json = json.dumps(fields, ensure_ascii=False, indent=2)
        
        # Get listing type specific instructions
        listing_type_instructions = self._get_listing_type_instructions(listing_type)
        
        # Define general translation guidelines
        general_guidelines = [
            f"Preserve the exact meaning and intent of the original text",
            f"Maintain any formatting (bullet points, line breaks, special characters)",
            f"Keep {listing_type.lower()}-specific terminology accurate and appropriate for the target language market",
            f"Do NOT translate brand names (e.g., Apple, Samsung, Nike) or model numbers that are part of product names (e.g., 'iPhone 15' - keep '15' as is, 'Galaxy S24' - keep '24' as is)",
            f"DO translate technology names and common technical terms to their standard translations in the target language (e.g., Bluetooth, WiFi, GPS, USB, HDMI, NFC, etc.)",
            f"IMPORTANT: For quantities, measurements, durations, and counts (e.g., '30 hours', '5 items', '10 meters', '3 pieces'), convert numbers to the appropriate numeral system used in the target language (e.g., Eastern Arabic numerals for Arabic, Devanagari numerals for Hindi, etc.)",
            f"For technical specifications with storage/memory sizes (e.g., '256GB', '8GB RAM'), you may keep the numbers in Western Arabic numerals (0-9) as they are commonly used in technical contexts, but translate the descriptive text",
            f"Adapt measurements, sizes, or formats if culturally appropriate (but keep values accurate)",
            f"Preserve any emojis or special formatting in the text",
            f"If the text contains technical terms, use the commonly accepted translation in the target market",
            f"Maintain the tone and style (formal/informal) appropriate for {listing_type.lower()} listings",
            f"Do NOT add or remove information - translate exactly what is provided",
            f"Keep HTML or markdown formatting if present in the original text"
        ]
        
        # Build the merged TRANSLATION GUIDELINES section
        # Start with listing type specific instructions, then add general guidelines
        guideline_number = 1
        guidelines = []
        
        # Add listing type specific instructions first
        for instruction in listing_type_instructions:
            guidelines.append(f"{guideline_number}. {instruction}")
            guideline_number += 1
        
        # Add general translation guidelines
        for guideline in general_guidelines:
            guidelines.append(f"{guideline_number}. {guideline}")
            guideline_number += 1
        
        guidelines_text = "\n".join(guidelines)
        
        prompt = f"""You are an expert translator specializing in {listing_type.lower()} listings and descriptions.

Your task is to translate the following listing fields to {target_language}.
{context_section}
FIELDS TO TRANSLATE:
{fields_json}

TRANSLATION GUIDELINES:
{guidelines_text}

CRITICAL: 
- Respond ONLY with a valid JSON object
- Do not include any markdown formatting or explanations outside the JSON
- The response must contain a "translations" object with the same field names as the input

Response format:
{{
  "translations": {{
    "field_name_1": "translated text for field 1",
    "field_name_2": "translated text for field 2"
  }}
}}

Example input (translating to Arabic):
Fields: {{"title": "iPhone 15 Pro Max - Excellent Condition", "description": "Barely used iPhone with original accessories.\\n- 256GB Storage\\n- Battery Health: 98%"}}

Example output:
{{
  "translations": {{
    "title": "iPhone 15 Pro Max - حالة ممتازة",
    "description": "iPhone مستخدم بشكل طفيف مع الملحقات الأصلية.\\n- سعة التخزين: 256GB\\n- صحة البطارية: 98%"
  }}
}}

Now translate the provided fields to {target_language}:"""

        return prompt

    async def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding vector for the given text using qwen3-embedding-8b model.
        
        Args:
            text: The text to create an embedding for
            
        Returns:
            List of floats representing the embedding vector
        """
        logger.debug(f"Creating embedding for text (length: {len(text)} chars)")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "Ma3rood AI Agents Embedding Service"
        }
        
        payload = {
            "model": self.EMBEDDING_MODEL,
            "input": text
        }
        
        async with httpx.AsyncClient(timeout=180.0) as client:
            try:
                response = await client.post(
                    self.embedding_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                
                if "data" in result and len(result["data"]) > 0:
                    embedding = result["data"][0]["embedding"]
                    logger.debug(f"Created embedding with dimension: {len(embedding)}")
                    return embedding
                else:
                    logger.error("No embedding data in API response")
                    raise ValueError("No embedding data in API response")
                    
            except httpx.HTTPStatusError as e:
                error_detail = f"OpenRouter Embedding API error: {e.response.status_code}"
                logger.error(f"HTTP error from OpenRouter Embedding API: {error_detail}")
                if e.response.text:
                    try:
                        error_data = e.response.json()
                        error_detail = error_data.get("error", {}).get("message", error_detail)
                    except:
                        error_detail = f"{error_detail} - {e.response.text}"
                raise Exception(error_detail)
            except httpx.RequestError as e:
                error_detail = f"Request error: {str(e)}"
                logger.error(f"Request error to OpenRouter Embedding API: {str(e)}", exc_info=True)
                raise Exception(error_detail)

    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embedding vectors for multiple texts in one API call when supported,
        otherwise falls back to sequential calls.
        
        Args:
            texts: List of texts to create embeddings for
            
        Returns:
            List of embedding vectors (same order as input texts)
        """
        if not texts:
            return []
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "Ma3rood AI Agents Embedding Service"
        }
        
        payload = {
            "model": self.EMBEDDING_MODEL,
            "input": texts
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    self.embedding_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                
                if "data" in result and len(result["data"]) > 0:
                    # API returns embeddings in order; preserve order by index if present
                    data = result["data"]
                    embeddings = [None] * len(data)
                    for item in data:
                        idx = item.get("index", len(embeddings))
                        if idx < len(embeddings):
                            embeddings[idx] = item["embedding"]
                        else:
                            embeddings.append(item["embedding"])
                    embeddings = [e for e in embeddings if e is not None]
                    if len(embeddings) == len(texts):
                        logger.debug(f"Created {len(embeddings)} embeddings in batch")
                        return embeddings
                
                # Fallback: sequential calls
                logger.debug("Batch embedding not fully supported, falling back to sequential")
                return [await self.create_embedding(t) for t in texts]
                
            except (httpx.HTTPStatusError, (KeyError, IndexError)) as e:
                # Fallback to sequential on batch API errors
                logger.warning(f"Batch embedding failed ({e}), falling back to sequential")
                return [await self.create_embedding(t) for t in texts]

    async def generate_category_semantic_description(
        self, 
        category_path: str
    ) -> Dict[str, Any]:
        """
        Generate a semantic description and relevant attributes for a category.
        
        Args:
            category_path: The full category path (e.g., "Electronics > Phones > Smartphones")
            
        Returns:
            Dictionary with:
                - semantic_description: A detailed description of what products belong in this category
                - relevant_attributes: List of relevant attributes for products in this category
        """
        start_time = time.time()
        logger.info(f"Generating semantic description for category: {category_path}")
        
        prompt = self._build_category_semantic_prompt(category_path)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "Ma3rood AI Agents Category Embedding Service"
        }
        
        payload = {
            "model": self.TEXT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        async with httpx.AsyncClient(timeout=360.0) as client:
            try:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    
                    # Parse JSON from response
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                    
                    parsed = json.loads(content)
                    
                    time_taken = time.time() - start_time
                    logger.info(f"Generated semantic description for {category_path} in {time_taken:.2f}s")
                    
                    return {
                        "semantic_description": parsed.get("semantic_description", ""),
                        "relevant_attributes": parsed.get("relevant_attributes", [])
                    }
                else:
                    raise ValueError("No choices in API response")
                    
            except httpx.HTTPStatusError as e:
                error_detail = f"OpenRouter API error: {e.response.status_code}"
                logger.error(f"HTTP error generating category description: {error_detail}")
                raise Exception(error_detail)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response for category: {category_path}")
                raise Exception(f"Failed to parse category description: {str(e)}")
            except Exception as e:
                logger.error(f"Error generating category description: {str(e)}", exc_info=True)
                raise

    def _build_category_semantic_prompt(self, category_path: str) -> str:
        """Build prompt for generating category semantic description"""
        
        prompt = f"""You are an expert in product categorization for an online marketplace.

Given the following product category path, generate:
1. A semantic description that describes what types of products belong in this category
2. A list of relevant attributes that are typically associated with products in this category

Category Path: {category_path}

IMPORTANT GUIDELINES:
1. The semantic description should be 2-3 sentences that clearly describe:
   - What types of products belong in this category
   - Key characteristics or features of products in this category
   - Common use cases or target audience (if applicable)

2. The relevant attributes should be specific to this category type. Choose from attributes like:
   - Brand, Model, Color, Size, Weight, Material
   - Memory/RAM, Storage, Hard Drive Size, Processor, Cores
   - Screen Size, Resolution, Battery Capacity, Operating System
   - Network Type, Camera, Battery Life
   - Gender, Season, Year
   - Mileage, Fuel Type, Transmission
   - Dimensions, Condition Details, Warranty, Accessories Included
   - Or any other category-specific attributes

3. Only include attributes that are genuinely relevant to this specific category.
   For example:
   - For "Mobile phones", include: Brand, Model, Storage, RAM, Screen Size, Battery Capacity, Camera, Operating System
   - For "Clothing & Fashion > Shoes", include: Brand, Size, Color, Material, Gender
   - For "Antiques & collectables", include: Year, Material, Condition Details, Dimensions

Respond ONLY with a valid JSON object in the following format:
{{
  "semantic_description": "A detailed description of what products belong in this category...",
  "relevant_attributes": ["Attribute1", "Attribute2", "Attribute3", ...]
}}

Do not include any explanations or markdown formatting outside the JSON."""

        return prompt

    def _build_category_semantic_batch_prompt(self, category_paths: List[str]) -> str:
        """Build prompt for generating semantic descriptions for a batch of categories"""
        categories_list = "\n".join([f"- {i+1}. {path}" for i, path in enumerate(category_paths)])
        
        prompt = f"""You are an expert in product categorization for an online marketplace.

For EACH of the following product category paths, generate:
1. A semantic description that describes what types of products belong in that category
2. A list of relevant attributes that are typically associated with products in that category

Category paths:
{categories_list}

IMPORTANT GUIDELINES (apply to EACH category in the list):
1. The semantic description should be 2-3 sentences that clearly describe:
   - What types of products belong in this category
   - Key characteristics or features of products in this category
   - Common use cases or target audience (if applicable)

2. The relevant attributes should be specific to each category type. Choose from attributes like:
   - Brand, Model, Color, Size, Weight, Material
   - Memory/RAM, Storage, Hard Drive Size, Processor, Cores
   - Screen Size, Resolution, Battery Capacity, Operating System
   - Network Type, Camera, Battery Life
   - Gender, Season, Year
   - Mileage, Fuel Type, Transmission
   - Dimensions, Condition Details, Warranty, Accessories Included
   - Or any other category-specific attributes

3. Only include attributes that are genuinely relevant to each specific category.
   For example:
   - For "Mobile phones", include: Brand, Model, Storage, RAM, Screen Size, Battery Capacity, Camera, Operating System
   - For "Clothing & Fashion > Shoes", include: Brand, Size, Color, Material, Gender
   - For "Antiques & collectables", include: Year, Material, Condition Details, Dimensions

4. Respond with a JSON array. Each element must have "semantic_description" and "relevant_attributes" (array of strings).
   The order of the array MUST match the order of the category paths above (first category = first object, etc.).

Respond ONLY with a valid JSON array, no other text or markdown:
[
  {{"semantic_description": "A detailed description of what products belong in this category...", "relevant_attributes": ["Attribute1", "Attribute2", ...]}},
  {{"semantic_description": "...", "relevant_attributes": ["Attribute1", ...]}},
  ...
]"""

        return prompt

    async def generate_category_semantic_descriptions_batch(
        self,
        category_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate semantic descriptions and relevant attributes for a batch of categories in one LLM call.
        On JSON parse failure (e.g. truncation), retries by splitting the batch or falling back to single-category calls.
        
        Args:
            category_paths: List of full category paths
            
        Returns:
            List of dicts with semantic_description and relevant_attributes (same order as input)
        """
        if not category_paths:
            return []
        
        try:
            return await self._generate_category_semantic_descriptions_batch_impl(category_paths)
        except json.JSONDecodeError as e:
            logger.warning(f"Batch JSON parse failed ({e}), retrying with smaller batches")
            if len(category_paths) == 1:
                result = await self.generate_category_semantic_description(category_paths[0])
                return [result]
            mid = len(category_paths) // 2
            first = await self.generate_category_semantic_descriptions_batch(category_paths[:mid])
            second = await self.generate_category_semantic_descriptions_batch(category_paths[mid:])
            return first + second

    async def _generate_category_semantic_descriptions_batch_impl(
        self,
        category_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """Internal: one LLM call for the batch. Raises JSONDecodeError on parse failure."""
        if not category_paths:
            return []
        
        start_time = time.time()
        logger.info(f"Generating semantic descriptions for batch of {len(category_paths)} categories")
        
        prompt = self._build_category_semantic_batch_prompt(category_paths)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "Ma3rood AI Agents Category Embedding Service"
        }
        
        payload = {
            "model": self.TEXT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 12000
        }
        
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                self.base_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            if "choices" not in result or len(result["choices"]) == 0:
                raise ValueError("No choices in API response")
            
            content = result["choices"][0]["message"]["content"]
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            content = _fix_trailing_comma_json(content)
            
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                content_retry = re.sub(r"}\s*(?:\r?\n\s*)+{", "},\n{", content)
                content_retry = re.sub(r"}\s+{", "},{", content_retry)
                parsed = json.loads(content_retry)
            
            if not isinstance(parsed, list):
                parsed = [parsed]
            
            time_taken = time.time() - start_time
            logger.info(f"Generated {len(parsed)} semantic descriptions in {time_taken:.2f}s")
            
            return [
                {
                    "semantic_description": item.get("semantic_description", ""),
                    "relevant_attributes": item.get("relevant_attributes", [])
                }
                for item in parsed
            ]

    async def select_category_and_generate_output(
        self,
        image_urls: List[str],
        visual_description: str,
        candidate_categories: List[Dict[str, Any]],
        language: str
    ) -> Dict[str, Any]:
        """
        Select the best category from candidates and generate final output.
        
        This is Step 3 of the image-to-form flow.
        
        Args:
            image_urls: List of URLs of product images
            visual_description: The visual description from Step 1
            candidate_categories: List of candidate categories with their attributes from Step 2
            language: Target language for the output
            
        Returns:
            Dictionary with:
                - description: Product description in target language
                - category: Selected category (id_path and category_path)
                - condition: Product condition
                - attributes: Extracted attribute values
        """
        start_time = time.time()
        logger.info(
            f"Selecting category and generating output - "
            f"Candidates: {len(candidate_categories)}, Language: {language}"
        )
        
        analytics_service = get_analytics_service()
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cached_tokens = 0
        reasoning_tokens = 0
        cost = 0.0
        
        prompt = self._build_category_selection_prompt(
            visual_description=visual_description,
            candidate_categories=candidate_categories,
            language=language,
            image_count=len(image_urls)
        )
        input_char_count = len(prompt)
        
        # Prepare content with images
        content = [{"type": "text", "text": prompt}]
        for image_url in image_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "Ma3rood AI Agents Category Selection"
        }
        
        payload = {
            "model": self.MODEL,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.3,
            "max_tokens": 3000,
            "usage": {"include": True}
        }
        
        async with httpx.AsyncClient(timeout=90.0) as client:
            try:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Extract token usage
                if "usage" in result:
                    usage = result["usage"]
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    cost = usage.get("cost", 0)
                    prompt_details = usage.get("prompt_tokens_details", {})
                    completion_details = usage.get("completion_tokens_details", {})
                    cached_tokens = prompt_details.get("cached_tokens", 0)
                    reasoning_tokens = completion_details.get("reasoning_tokens", 0)
                
                if "choices" in result and len(result["choices"]) > 0:
                    content_text = result["choices"][0]["message"]["content"]
                    
                    # Parse JSON from response
                    content_text = content_text.strip()
                    if content_text.startswith("```json"):
                        content_text = content_text[7:]
                    if content_text.startswith("```"):
                        content_text = content_text[3:]
                    if content_text.endswith("```"):
                        content_text = content_text[:-3]
                    content_text = content_text.strip()
                    
                    parsed = json.loads(content_text)
                    output_char_count = len(json.dumps(parsed, ensure_ascii=False))
                    
                    time_taken = time.time() - start_time
                    logger.info(
                        f"Category selection completed in {time_taken:.2f}s - "
                        f"Selected: {parsed.get('category', {}).get('category_path', 'N/A')}"
                    )
                    
                    # Log analytics
                    analytics_service.log_analytics(
                        agent_type="image_to_form_v2",
                        model=self.MODEL,
                        provider="OpenRouter",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        cached_tokens=cached_tokens,
                        reasoning_tokens=reasoning_tokens,
                        cost=cost,
                        input_char_count=input_char_count,
                        output_char_count=output_char_count,
                        time_taken_seconds=time_taken,
                        target_language=language,
                        status="success",
                        error_message=None
                    )
                    
                    return parsed
                else:
                    raise ValueError("No choices in API response")
                    
            except httpx.HTTPStatusError as e:
                error_detail = f"OpenRouter API error: {e.response.status_code}"
                logger.error(f"HTTP error in category selection: {error_detail}")
                
                time_taken = time.time() - start_time
                analytics_service.log_analytics(
                    agent_type="image_to_form_v2",
                    model=self.MODEL,
                    provider="OpenRouter",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cached_tokens=cached_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cost=cost,
                    input_char_count=input_char_count,
                    output_char_count=0,
                    time_taken_seconds=time_taken,
                    target_language=language,
                    status="error",
                    error_message=error_detail
                )
                
                raise Exception(error_detail)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response in category selection: {str(e)}")
                raise Exception(f"Failed to parse category selection response: {str(e)}")
            except Exception as e:
                logger.error(f"Error in category selection: {str(e)}", exc_info=True)
                raise

    def _build_category_selection_prompt(
        self,
        visual_description: str,
        candidate_categories: List[Dict[str, Any]],
        language: str,
        image_count: int = 1
    ) -> str:
        """Build prompt for category selection and output generation (aligned with marketplace prompt)."""
        
        image_text = "these images" if image_count > 1 else "this image"
        image_instruction = f"Analyze all {image_count} images provided" if image_count > 1 else "Analyze this image"
        
        categories_section = ""
        for i, cat in enumerate(candidate_categories, 1):
            attrs = ", ".join(cat.get("relevant_attributes", []))
            categories_section += f"""
Category {i}:
- ID Path: {cat.get('id_path', '')}
- Category Path: {cat.get('category_path', '')}
- Relevant Attributes for this category: {attrs}
"""
        
        prompt = f"""{image_instruction} and extract all relevant information for a marketplace listing form.
All {image_text} show the same product from different angles or views. Combine information from all images to get a complete picture of the product.

FIRST: Select the BEST matching category from the following CANDIDATE CATEGORIES. Use the selected category's "Relevant Attributes" list when filling the attributes field below.

CANDIDATE CATEGORIES (choose exactly one):
{categories_section}

VISUAL DESCRIPTION OF THE PRODUCT (for context):
{visual_description}

Respond ONLY with a valid JSON object in {language} language. Do not include any markdown formatting or explanations outside the JSON.

REQUIRED FIELDS (always include these):
- title: Product/item title (string, required) - generate from the images
- description: Detailed, marketing-friendly description of the product/item (string, required) - generate by looking at the images; make it compelling and appealing to buyers; highlight key benefits/features and condition
- category: The selected category ONLY - use this exact format: {{ "id_path": "<id_path of selected category>", "category_path": "<category_path of selected category>" }}
- condition: Condition of the item. Must be one of the following exact values (string, or null if not determinable):
  * "Brand New or Unused": never opened or used
  * "Like New": opened but looks and works like new
  * "Gently Used or Excellent Condition": minor signs of use
  * "Good Condition": visible wear but fully functional
  * "Fair Condition": heavily used but still works
  * "For Parts or Not Working": damaged or needs repair
  * "Not Applicable": condition does not apply to this item

ATTRIBUTES (include ONLY attributes from the selected category's "Relevant Attributes" list above):
Analyze the product in the images and extract ONLY the attributes that are in the selected category's relevant attributes list AND that are visible or determinable for this product.
You may also include any of these if they are in the selected category's list and relevant:
- Brand, Model, Color, Size, Weight, Material
- Memory/RAM, Storage, Hard Drive Size, Processor, Cores
- Screen Size, Resolution, Battery Capacity, Operating System
- Network Type, Camera, Battery Life
- Gender, Season, Year
- Mileage, Fuel Type, Transmission
- Dimensions, Condition Details, Warranty, Accessories Included

CRITICAL INSTRUCTIONS:
1. Analyze ALL provided images together to extract comprehensive information about the product
2. Generate the description by looking at the images - describe what you see: features, condition, key selling points
3. For the "condition" field, you MUST use one of the exact values listed above. Analyze the images carefully to determine the most appropriate condition based on visible wear, packaging, and usage signs
4. Include ONLY attributes that are in the selected category's relevant attributes list AND relevant to the product shown in the images
5. Do NOT include attributes that are not in the selected category's list or not applicable (e.g., don't include "Mileage" for a smartphone)
6. Attribute names should be in English (camelCase or Title Case)
7. Attribute values should be in {language} language (except for condition which must use the exact English values listed)
8. If an attribute is not visible or cannot be determined from any of the images, do NOT include it in the response
9. All text (title, description, tags, attribute values) should be in {language} language (except condition field)
10. Be accurate and only extract information that is clearly visible in the images
11. Use information from all images to build a complete product description

Response format:
{{
  "title": "Product Title",
  "description": "Detailed description generated from the images...",
  "category": {{ "id_path": "selected_id_path", "category_path": "Selected > Category > Path" }},
  "condition": "Brand New or Unused",
  "attributes": {{
    "Brand": "Brand Name",
    "Model": "Model XYZ",
    "Color": "Black"
  }}
}}

Example for a smartphone (category selected: Mobile phones > Smartphones):
{{
  "title": "iPhone 15 Pro",
  "description": "Latest iPhone with advanced features...",
  "category": {{ "id_path": "3638 > 3640", "category_path": "Mobile phones > Smartphones" }},
  "condition": "Brand New or Unused",
  "attributes": {{
    "Brand": "Apple",
    "Model": "iPhone 15 Pro",
    "Color": "Titanium Blue",
    "Storage": "256GB",
    "Screen Size": "6.1 inches",
    "Battery Capacity": "3274 mAh",
    "Operating System": "iOS 17"
  }}
}}

Example for clothing:
{{
  "title": "Nike Running Shoes",
  "description": "Comfortable running shoes...",
  "category": {{ "id_path": "1482 > 1500", "category_path": "Clothing & Fashion > Shoes" }},
  "condition": "Brand New or Unused",
  "attributes": {{
    "Brand": "Nike",
    "Color": "Black/White",
    "Size": "10",
    "Gender": "Unisex",
    "Material": "Mesh and Synthetic"
  }}
}}

Do not include any explanations or markdown formatting outside the JSON."""

        return prompt

    async def extract_visual_description_and_root_category(
        self,
        image_urls: List[str]
    ) -> Dict[str, Any]:
        """
        Extract a factual visual description and determine the root category from product images.
        
        This is Step 1 of the image-to-form flow.
        
        Args:
            image_urls: List of URLs of product images
            
        Returns:
            Dictionary with:
                - visual_description: Factual description of what's visible in the images
                - root_category: One of the 23 predefined root categories
        """
        start_time = time.time()
        logger.info(f"Extracting visual description and root category from {len(image_urls)} images")
        
        prompt = self._build_visual_extraction_prompt()
        
        # Prepare content with images
        content = [{"type": "text", "text": prompt}]
        for image_url in image_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "Ma3rood AI Agents Visual Extraction"
        }
        
        payload = {
            "model": self.MODEL,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.3,
            "max_tokens": 1500
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    content_text = result["choices"][0]["message"]["content"]
                    
                    # Parse JSON from response
                    content_text = content_text.strip()
                    if content_text.startswith("```json"):
                        content_text = content_text[7:]
                    if content_text.startswith("```"):
                        content_text = content_text[3:]
                    if content_text.endswith("```"):
                        content_text = content_text[:-3]
                    content_text = content_text.strip()
                    
                    parsed = json.loads(content_text)
                    
                    # Validate root category
                    root_category = parsed.get("root_category", "")
                    if root_category not in self.ROOT_CATEGORIES:
                        # Try to find closest match
                        root_category_lower = root_category.lower()
                        for valid_category in self.ROOT_CATEGORIES:
                            if valid_category.lower() in root_category_lower or root_category_lower in valid_category.lower():
                                root_category = valid_category
                                break
                        else:
                            # Default to most generic category if no match
                            logger.warning(f"Invalid root category '{parsed.get('root_category')}', defaulting to first match")
                            root_category = self.ROOT_CATEGORIES[0]
                    
                    time_taken = time.time() - start_time
                    logger.info(
                        f"Visual extraction completed in {time_taken:.2f}s - "
                        f"Root category: {root_category}"
                    )
                    
                    return {
                        "visual_description": parsed.get("visual_description", ""),
                        "root_category": root_category
                    }
                else:
                    raise ValueError("No choices in API response")
                    
            except httpx.HTTPStatusError as e:
                error_detail = f"OpenRouter API error: {e.response.status_code}"
                logger.error(f"HTTP error in visual extraction: {error_detail}")
                raise Exception(error_detail)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response in visual extraction: {str(e)}")
                raise Exception(f"Failed to parse visual extraction response: {str(e)}")
            except Exception as e:
                logger.error(f"Error in visual extraction: {str(e)}", exc_info=True)
                raise

    def _build_visual_extraction_prompt(self) -> str:
        """Build prompt for extracting visual description and root category"""
        
        root_categories_list = "\n".join([f"- {cat}" for cat in self.ROOT_CATEGORIES])
        
        prompt = f"""Analyze the product image(s) and provide:

1. A factual visual description of the product - describe exactly what you see, including:
   - The type/kind of product
   - Physical characteristics (color, shape, size if apparent)
   - Brand name or logos if visible
   - Model information if visible
   - Condition indicators (new in box, used, damaged, etc.)
   - Any text, labels, or markings visible
   - Accessories or items included
   - Material if determinable

2. The root category that best matches this product from the following list:

{root_categories_list}

IMPORTANT:
- Be objective and factual in the description
- Only describe what is actually visible in the image(s)
- Do not make assumptions about features you cannot see
- The visual description should be detailed enough to enable finding the right product category
- Choose exactly ONE root category from the list above

Respond ONLY with a valid JSON object:
{{
  "visual_description": "A detailed, factual description of what is visible in the image(s)...",
  "root_category": "Exact category name from the list above"
}}

Do not include any explanations or markdown formatting outside the JSON."""

        return prompt
