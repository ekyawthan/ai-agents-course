# Multimodal Agents

## Introduction to Multimodal AI

Multimodal agents can process and generate multiple types of data: text, images, audio, video, and more. This enables richer interactions and broader capabilities.

### Why Multimodal Matters

**Benefits**:
- Richer understanding of context
- More natural interactions
- Broader range of tasks
- Better accessibility
- Cross-modal reasoning

**Challenges**:
- Increased complexity
- Higher computational costs
- Data alignment across modalities
- Quality control
- Privacy concerns

### Modalities

1. **Vision**: Images, videos, screenshots
2. **Audio**: Speech, music, sounds
3. **Text**: Natural language
4. **Documents**: PDFs, spreadsheets
5. **Structured Data**: Tables, graphs

## Vision and Image Understanding

### Image Analysis

```python
import base64
from pathlib import Path
import openai

class VisionAgent:
    """Agent with vision capabilities"""
    
    def __init__(self):
        self.client = openai.OpenAI()
    
    def analyze_image(self, image_path: str, question: str = None) -> str:
        """Analyze image and answer questions"""
        
        # Read and encode image
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Determine image type
        ext = Path(image_path).suffix.lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }.get(ext, 'image/jpeg')
        
        # Build prompt
        if question:
            prompt = question
        else:
            prompt = "Describe this image in detail."
        
        # Call vision model
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image (OCR)"""
        return self.analyze_image(
            image_path,
            "Extract all text from this image. Provide the text exactly as it appears."
        )
    
    def describe_scene(self, image_path: str) -> Dict:
        """Get detailed scene description"""
        description = self.analyze_image(
            image_path,
            """Describe this image in detail:
            1. Main subjects
            2. Setting/location
            3. Actions/activities
            4. Colors and mood
            5. Notable details"""
        )
        
        return {"description": description}
    
    def identify_objects(self, image_path: str) -> List[str]:
        """Identify objects in image"""
        result = self.analyze_image(
            image_path,
            "List all objects visible in this image, one per line."
        )
        
        # Parse list
        objects = [line.strip('- ').strip() for line in result.split('\n') if line.strip()]
        return objects
    
    def compare_images(self, image1_path: str, image2_path: str) -> str:
        """Compare two images"""
        
        # Encode both images
        images_data = []
        for path in [image1_path, image2_path]:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode('utf-8')
                images_data.append(data)
        
        # Compare
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these two images. What are the similarities and differences?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{images_data[0]}"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{images_data[1]}"}
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def answer_visual_question(self, image_path: str, question: str) -> str:
        """Answer specific question about image"""
        return self.analyze_image(image_path, question)

# Usage
vision_agent = VisionAgent()

# Analyze image
description = vision_agent.analyze_image("photo.jpg")
print(f"Description: {description}")

# Extract text (OCR)
text = vision_agent.extract_text_from_image("document.jpg")
print(f"Extracted text: {text}")

# Identify objects
objects = vision_agent.identify_objects("scene.jpg")
print(f"Objects: {objects}")

# Answer question
answer = vision_agent.answer_visual_question(
    "chart.jpg",
    "What is the trend shown in this chart?"
)
print(f"Answer: {answer}")
```

### Image Generation

```python
class ImageGenerator:
    """Generate images from text"""
    
    def __init__(self):
        self.client = openai.OpenAI()
    
    def generate_image(self, 
                      prompt: str,
                      size: str = "1024x1024",
                      quality: str = "standard",
                      n: int = 1) -> List[str]:
        """Generate image from text prompt"""
        
        response = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            n=n
        )
        
        # Get URLs
        image_urls = [img.url for img in response.data]
        
        return image_urls
    
    def edit_image(self, 
                   image_path: str,
                   mask_path: str,
                   prompt: str) -> str:
        """Edit image using mask"""
        
        response = self.client.images.edit(
            image=open(image_path, "rb"),
            mask=open(mask_path, "rb"),
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        
        return response.data[0].url
    
    def create_variation(self, image_path: str, n: int = 1) -> List[str]:
        """Create variations of image"""
        
        response = self.client.images.create_variation(
            image=open(image_path, "rb"),
            n=n,
            size="1024x1024"
        )
        
        return [img.url for img in response.data]

# Usage
generator = ImageGenerator()

# Generate image
urls = generator.generate_image(
    "A futuristic AI agent helping humans",
    quality="hd"
)
print(f"Generated: {urls[0]}")

# Create variations
variations = generator.create_variation("original.png", n=3)
print(f"Created {len(variations)} variations")
```

## Audio Processing

### Speech Recognition

```python
class AudioAgent:
    """Agent with audio capabilities"""
    
    def __init__(self):
        self.client = openai.OpenAI()
    
    def transcribe_audio(self, audio_path: str, language: str = None) -> Dict:
        """Transcribe audio to text"""
        
        with open(audio_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="verbose_json"
            )
        
        return {
            "text": transcript.text,
            "language": transcript.language,
            "duration": transcript.duration,
            "segments": transcript.segments if hasattr(transcript, 'segments') else []
        }
    
    def translate_audio(self, audio_path: str) -> str:
        """Translate audio to English"""
        
        with open(audio_path, "rb") as audio_file:
            translation = self.client.audio.translations.create(
                model="whisper-1",
                file=audio_file
            )
        
        return translation.text
    
    def transcribe_with_timestamps(self, audio_path: str) -> List[Dict]:
        """Transcribe with word-level timestamps"""
        
        result = self.transcribe_audio(audio_path)
        
        segments = []
        for segment in result.get("segments", []):
            segments.append({
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text")
            })
        
        return segments

# Usage
audio_agent = AudioAgent()

# Transcribe
result = audio_agent.transcribe_audio("speech.mp3")
print(f"Transcription: {result['text']}")
print(f"Language: {result['language']}")

# Translate
translation = audio_agent.translate_audio("french_audio.mp3")
print(f"Translation: {translation}")

# With timestamps
segments = audio_agent.transcribe_with_timestamps("interview.mp3")
for seg in segments:
    print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}")
```

### Text-to-Speech

```python
class TextToSpeech:
    """Convert text to speech"""
    
    def __init__(self):
        self.client = openai.OpenAI()
    
    def synthesize_speech(self,
                         text: str,
                         voice: str = "alloy",
                         model: str = "tts-1",
                         output_path: str = "speech.mp3") -> str:
        """Convert text to speech
        
        Voices: alloy, echo, fable, onyx, nova, shimmer
        Models: tts-1 (faster), tts-1-hd (higher quality)
        """
        
        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text
        )
        
        # Save to file
        response.stream_to_file(output_path)
        
        return output_path
    
    def synthesize_long_text(self,
                            text: str,
                            voice: str = "alloy",
                            chunk_size: int = 4000) -> List[str]:
        """Synthesize long text in chunks"""
        
        # Split into chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        output_files = []
        for i, chunk in enumerate(chunks):
            output_path = f"speech_part_{i}.mp3"
            self.synthesize_speech(chunk, voice, output_path=output_path)
            output_files.append(output_path)
        
        return output_files

# Usage
tts = TextToSpeech()

# Synthesize
audio_file = tts.synthesize_speech(
    "Hello! I am an AI agent with voice capabilities.",
    voice="nova"
)
print(f"Generated audio: {audio_file}")
```

## Document Parsing

### PDF Processing

```python
import PyPDF2
from typing import List, Dict

class DocumentAgent:
    """Process various document types"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.vision_agent = VisionAgent()
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """Extract text from PDF"""
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            text_by_page = []
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                text_by_page.append({
                    "page": page_num + 1,
                    "text": text
                })
            
            full_text = "\n\n".join([p["text"] for p in text_by_page])
            
            return {
                "num_pages": len(pdf_reader.pages),
                "pages": text_by_page,
                "full_text": full_text
            }
    
    def analyze_pdf_with_vision(self, pdf_path: str) -> List[Dict]:
        """Analyze PDF pages as images"""
        
        # Convert PDF pages to images (requires pdf2image)
        from pdf2image import convert_from_path
        
        images = convert_from_path(pdf_path)
        
        analyses = []
        for i, image in enumerate(images):
            # Save temporarily
            temp_path = f"temp_page_{i}.jpg"
            image.save(temp_path, 'JPEG')
            
            # Analyze with vision
            analysis = self.vision_agent.analyze_image(temp_path)
            
            analyses.append({
                "page": i + 1,
                "analysis": analysis
            })
            
            # Clean up
            import os
            os.remove(temp_path)
        
        return analyses
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract tables from PDF"""
        
        # Using tabula-py for table extraction
        import tabula
        
        tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
        
        extracted = []
        for i, table in enumerate(tables):
            extracted.append({
                "table_num": i + 1,
                "data": table.to_dict('records'),
                "shape": table.shape
            })
        
        return extracted
    
    def summarize_document(self, text: str, max_length: int = 500) -> str:
        """Summarize document"""
        
        prompt = f"""Summarize this document in {max_length} words or less:

{text[:10000]}  # Limit input

Summary:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def answer_document_question(self, text: str, question: str) -> str:
        """Answer question about document"""
        
        prompt = f"""Based on this document, answer the question:

Document:
{text[:8000]}

Question: {question}

Answer:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content

# Usage
doc_agent = DocumentAgent()

# Extract text
result = doc_agent.extract_text_from_pdf("document.pdf")
print(f"Pages: {result['num_pages']}")
print(f"First page: {result['pages'][0]['text'][:200]}...")

# Summarize
summary = doc_agent.summarize_document(result['full_text'])
print(f"Summary: {summary}")

# Answer question
answer = doc_agent.answer_document_question(
    result['full_text'],
    "What are the main conclusions?"
)
print(f"Answer: {answer}")
```

## Cross-Modal Reasoning

### Multimodal Understanding

```python
class MultimodalAgent:
    """Agent that reasons across modalities"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.vision = VisionAgent()
        self.audio = AudioAgent()
        self.document = DocumentAgent()
    
    def analyze_multimodal_input(self, inputs: Dict) -> str:
        """Analyze multiple types of input together"""
        
        context = "Analyzing multimodal input:\n\n"
        
        # Process each modality
        if "image" in inputs:
            image_analysis = self.vision.analyze_image(inputs["image"])
            context += f"Image: {image_analysis}\n\n"
        
        if "audio" in inputs:
            audio_transcript = self.audio.transcribe_audio(inputs["audio"])
            context += f"Audio: {audio_transcript['text']}\n\n"
        
        if "text" in inputs:
            context += f"Text: {inputs['text']}\n\n"
        
        if "document" in inputs:
            doc_content = self.document.extract_text_from_pdf(inputs["document"])
            context += f"Document: {doc_content['full_text'][:1000]}...\n\n"
        
        # Synthesize understanding
        prompt = f"""{context}

Based on all this information, provide a comprehensive analysis:
1. Key themes across all modalities
2. How the different inputs relate to each other
3. Overall insights

Analysis:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        return response.choices[0].message.content
    
    def generate_multimodal_response(self, 
                                    query: str,
                                    include_image: bool = False,
                                    include_audio: bool = False) -> Dict:
        """Generate response in multiple modalities"""
        
        # Generate text response
        text_response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": query}]
        ).choices[0].message.content
        
        result = {"text": text_response}
        
        # Generate image if requested
        if include_image:
            # Extract visual description from text
            image_prompt = self.extract_visual_description(text_response)
            generator = ImageGenerator()
            image_url = generator.generate_image(image_prompt)[0]
            result["image"] = image_url
        
        # Generate audio if requested
        if include_audio:
            tts = TextToSpeech()
            audio_file = tts.synthesize_speech(text_response)
            result["audio"] = audio_file
        
        return result
    
    def extract_visual_description(self, text: str) -> str:
        """Extract visual description for image generation"""
        
        prompt = f"""From this text, create a detailed visual description suitable for image generation:

{text}

Visual description:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        return response.choices[0].message.content
    
    def create_presentation(self, topic: str, num_slides: int = 5) -> List[Dict]:
        """Create multimodal presentation"""
        
        # Generate outline
        outline_prompt = f"Create a {num_slides}-slide presentation outline about: {topic}"
        
        outline_response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": outline_prompt}]
        )
        
        outline = outline_response.choices[0].message.content
        
        # Generate each slide
        slides = []
        generator = ImageGenerator()
        tts = TextToSpeech()
        
        for i in range(num_slides):
            # Generate slide content
            slide_prompt = f"""Create content for slide {i+1} of presentation about {topic}.
            
Outline: {outline}

Provide:
1. Title
2. Key points (3-5 bullets)
3. Visual description for image

Slide content:"""
            
            slide_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": slide_prompt}]
            )
            
            slide_content = slide_response.choices[0].message.content
            
            # Generate image
            visual_desc = self.extract_visual_description(slide_content)
            image_url = generator.generate_image(visual_desc)[0]
            
            # Generate narration audio
            audio_file = tts.synthesize_speech(
                slide_content,
                output_path=f"slide_{i+1}_narration.mp3"
            )
            
            slides.append({
                "slide_num": i + 1,
                "content": slide_content,
                "image": image_url,
                "audio": audio_file
            })
        
        return slides

# Usage
multimodal_agent = MultimodalAgent()

# Analyze multimodal input
analysis = multimodal_agent.analyze_multimodal_input({
    "image": "chart.jpg",
    "text": "This shows our quarterly results",
    "audio": "explanation.mp3"
})
print(f"Analysis: {analysis}")

# Generate multimodal response
response = multimodal_agent.generate_multimodal_response(
    "Explain quantum computing",
    include_image=True,
    include_audio=True
)
print(f"Text: {response['text']}")
print(f"Image: {response['image']}")
print(f"Audio: {response['audio']}")

# Create presentation
slides = multimodal_agent.create_presentation("AI Agents", num_slides=3)
for slide in slides:
    print(f"Slide {slide['slide_num']}: {slide['content'][:100]}...")
```

## Best Practices

1. **Choose right modality**: Use most appropriate for task
2. **Quality control**: Validate outputs across modalities
3. **Accessibility**: Provide alternatives (captions, transcripts)
4. **Privacy**: Handle sensitive data carefully
5. **Cost management**: Multimodal can be expensive
6. **Caching**: Reuse processed results
7. **Error handling**: Each modality can fail differently
8. **User preferences**: Let users choose modalities
9. **Testing**: Test across all modalities
10. **Performance**: Optimize processing pipelines

## Next Steps

You now understand multimodal agents in depth! Next, we'll explore agentic frameworks that help build complex agent systems.
