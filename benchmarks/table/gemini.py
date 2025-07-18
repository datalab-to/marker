import json
from PIL import Image
from google import genai
from google.genai import types
from io import BytesIO
from pydantic import BaseModel

from marker.settings import settings

prompt = """
You're an expert document analyst who is good at turning tables in documents into HTML.  Analyze the provided image, and convert it to a faithful HTML representation.
 
Guidelines:
- Keep the HTML simple and concise.
- Only include the <table> tag and contents.
- Only use <table>, <tr>, and <td> tags.  Only use the colspan and rowspan attributes if necessary.  Do not use <tbody>, <thead>, or <th> tags.
- Make sure the table is as faithful to the image as possible with the given tags.

**Instructions**
1. Analyze the image, and determine the table structure.
2. Convert the table image to HTML, following the guidelines above.
3. Output only the HTML for the table, starting with the <table> tag and ending with the </table> tag.
""".strip()

class TableSchema(BaseModel):
    table_html: str

def gemini_table_rec(image: Image.Image):
    client = genai.Client(
        api_key=settings.GOOGLE_API_KEY,
        http_options={"timeout": 60000}
    )

    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")

    responses = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Part.from_bytes(data=image_bytes.getvalue(), mime_type="image/png"), prompt],  # According to gemini docs, it performs better if the image is the first element
        config={
            "temperature": 0,
            "response_schema": TableSchema,
            "response_mime_type": "application/json",
        },
    )

    output = responses.candidates[0].content.parts[0].text
    return json.loads(output)["table_html"]