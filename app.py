"""
Photo Number Extractor
Aplicación web para extraer números de fotos usando Groq Vision API (Llama 3.2 Vision)
"""

import os
import base64
from flask import Flask, render_template, request, jsonify
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB max

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
client = Groq(api_key=GROQ_API_KEY)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_numbers_from_image(image_base64, mime_type="image/jpeg"):
    """Envía la imagen a Groq Vision API y extrae los números."""
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sos un asistente experto en leer y extraer números de imágenes. "
                        "Tu tarea es identificar TODOS los números visibles en la imagen. "
                        "Incluí números de factura, fechas, montos, códigos, cantidades, teléfonos, etc. "
                        "Respondé SOLO en formato JSON con la siguiente estructura:\n"
                        '{"numeros": [{"valor": "12345", "contexto": "Número de factura"}, '
                        '{"valor": "1500.50", "contexto": "Monto total"}]}\n'
                        "Si no encontrás números, respondé: {\"numeros\": []}\n"
                        "IMPORTANTE: Respondé ÚNICAMENTE con el JSON, sin texto adicional."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extraé todos los números que veas en esta imagen. Incluí su contexto (qué representan)."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=2048,
            response_format={"type": "json_object"}
        )
        return {
            "success": True,
            "data": response.choices[0].message.content,
            "model": response.model,
            "tokens": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.route('/sw.js')
def service_worker():
    """Serve service worker from root scope."""
    return app.send_static_file('sw.js'), 200, {'Content-Type': 'application/javascript'}


@app.route('/ping')
def ping():
    """Keepalive endpoint for cron jobs."""
    return jsonify({"status": "ok"})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/extract', methods=['POST'])
def extract():
    """Endpoint para extraer números de una imagen subida."""
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No se envió ninguna imagen"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"success": False, "error": "No se seleccionó ningún archivo"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Formato no soportado. Usá: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # Leer y convertir a base64
    image_bytes = file.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # Determinar MIME type
    ext = file.filename.rsplit('.', 1)[1].lower()
    mime_map = {
        'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png',
        'gif': 'image/gif', 'bmp': 'image/bmp', 'webp': 'image/webp'
    }
    mime_type = mime_map.get(ext, 'image/jpeg')

    # Extraer números
    result = extract_numbers_from_image(image_base64, mime_type)
    return jsonify(result)


@app.route('/extract-base64', methods=['POST'])
def extract_base64():
    """Endpoint para extraer números de una imagen en base64 (desde cámara)."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"success": False, "error": "No se envió ninguna imagen"}), 400

    # La imagen viene como data:image/jpeg;base64,xxxx
    image_data = data['image']
    if ',' in image_data:
        header, image_base64 = image_data.split(',', 1)
        mime_type = header.split(':')[1].split(';')[0] if ':' in header else 'image/jpeg'
    else:
        image_base64 = image_data
        mime_type = 'image/jpeg'

    result = extract_numbers_from_image(image_base64, mime_type)
    return jsonify(result)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  📸 Photo Number Extractor")
    print("  Abrí http://localhost:5015 en tu navegador")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5015, debug=True)
