import httpx
import json
import time
from typing import Dict, Any, Optional, List
from app.core.config import settings
from app.core.logging_config import get_logger
from app.core.analytics import get_analytics_service

logger = get_logger(__name__)


class OpenRouterService:
    """Service for interacting with OpenRouter API using Qwen3 VL model"""
    
    MODEL = "qwen/qwen3-vl-235b-a22b-instruct"  # Vision model for image processing
    TEXT_MODEL = "qwen/qwen3-235b-a22b"  # Text model for translation and text tasks
    
    def __init__(self):
        self.api_key = settings.OPENROUTER_API_KEY
        if not self.api_key:
            logger.error("OPENROUTER_API_KEY is not set in environment variables")
            raise ValueError("OPENROUTER_API_KEY is not set in environment variables")
        self.base_url = settings.OPENROUTER_BASE_URL
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
- description: Detailed description of the product/item (string, required)
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
