import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from router import PipelineRouter
from face_model import build_deepfake_detector
from general_model import build_general_ai_detector
import matplotlib.cm as cm

app = Flask(__name__)

print("Booting up Advanced Detection Pipeline...")
router = PipelineRouter()

print("Building Fresh Face Model & Injecting Weights...")
face_model = build_deepfake_detector(input_shape=(224, 224, 3))
face_model.load_weights('best_deepfake_model.weights.h5')

print("Building Fresh General Model & Injecting Weights...")
general_model = build_general_ai_detector(input_shape=(224, 224, 3))
general_model.load_weights('best_general_model.weights.h5')

print("System Ready! All modules online.")

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# --- GRAD-CAM GENERATOR ---
def generate_gradcam(img_array, full_model, original_rgb_img):
    """Slices the model in half to track gradients and generate a thermal heatmap."""
    # 1. Split the model (Base Extractor vs. Classifier Head)
    base_model = full_model.layers[1]
    
    feature_map_model = tf.keras.models.Model(
        inputs=full_model.inputs,
        outputs=base_model(full_model.inputs)
    )
    
    classifier_input = tf.keras.Input(shape=base_model.output_shape[1:])
    x = classifier_input
    for layer in full_model.layers[2:]:
        x = layer(x)
    classifier_model = tf.keras.models.Model(inputs=classifier_input, outputs=x)
    
    # 2. Track the math (Gradients)
    with tf.GradientTape() as tape:
        feature_maps = feature_map_model(img_array)
        tape.watch(feature_maps)
        preds = classifier_model(feature_maps)
        score = preds[0][0]
        
        # If fake, track what makes it fake. If real, track what makes it real.
        class_channel = 1.0 - score if score < 0.5 else score

    # 3. Calculate importance weights
    grads = tape.gradient(class_channel, feature_maps)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    feature_maps = feature_maps[0]
    heatmap = feature_maps @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0)
    max_heat = tf.math.reduce_max(heatmap)
    if max_heat == 0: max_heat = 1e-10
    heatmap = heatmap / max_heat
    heatmap = heatmap.numpy()
    
    # 4. Colorize and overlay
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    jet_heatmap = cv2.resize(jet_heatmap, (original_rgb_img.shape[1], original_rgb_img.shape[0]))
    jet_heatmap = np.uint8(255 * jet_heatmap)
    
    # Blend the images (60% original, 40% thermal map)
    superimposed_img = cv2.addWeighted(original_rgb_img, 0.6, jet_heatmap, 0.4, 0)
    return superimposed_img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        route_data = router.route_image(filepath)
        heatmap_filename = f"heatmap_{file.filename}.jpg"
        heatmap_path = os.path.join(STATIC_FOLDER, heatmap_filename)
        
        # BRANCH A: Deepfake Face Detection
        if route_data['route'] == 'BOTH' and route_data['cropped_faces']:
            face_img = route_data['cropped_faces'][0] 
            face_resized = cv2.resize(face_img, (224, 224))
            input_array = np.expand_dims(face_resized, axis=0).astype('float32')
            
            prediction = float(face_model.predict(input_array, verbose=0)[0][0])
            real_prob = round(prediction * 100, 2)
            fake_prob = round(100 - real_prob, 2)
            
            heatmap_img = generate_gradcam(input_array, face_model, face_img)
            cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))
            
            # --- NEW: Dynamic Explanation ---
            if real_prob > 50:
                explanation = "🔴 <b class='text-red-400'>Red/Yellow Areas:</b> The AI focused here and found genuine photographic traits (like real skin pores, natural edge lighting, or correct physical geometry).<br>🔵 <b class='text-blue-400'>Blue Areas:</b> The AI mostly ignored these pixels."
            else:
                explanation = "🔴 <b class='text-red-400'>Red/Yellow Areas:</b> The AI focused here and detected strong manipulation artifacts (like AI blending errors, asymmetrical eyes, or synthetic noise).<br>🔵 <b class='text-blue-400'>Blue Areas:</b> The AI mostly ignored these pixels."

            result = {
                "branch_used": "Branch A (Facial Deepfake)",
                "status": "Face Detected",
                "fake_confidence": fake_prob,
                "real_confidence": real_prob,
                "verdict": "Real Photograph" if real_prob > 50 else "AI-Generated / Deepfake",
                "heatmap_url": f"/static/{heatmap_filename}",
                "explanation": explanation
            }
            
        # BRANCH B: General AI Art Detection
        else:
            full_img = route_data['full_image'] 
            img_resized = cv2.resize(full_img, (224, 224))
            input_array = np.expand_dims(img_resized, axis=0).astype('float32')
            
            prediction = float(general_model.predict(input_array, verbose=0)[0][0])
            real_prob = round(prediction * 100, 2)
            fake_prob = round(100 - real_prob, 2)
            
            heatmap_img = generate_gradcam(input_array, general_model, full_img)
            cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))
            
            # --- NEW: Dynamic Explanation ---
            if real_prob > 50:
                explanation = "🔴 <b class='text-red-400'>Red/Yellow Areas:</b> The AI verified natural camera lens properties and genuine textures here.<br>🔵 <b class='text-blue-400'>Blue Areas:</b> The AI considered these areas irrelevant."
            else:
                explanation = "🔴 <b class='text-red-400'>Red/Yellow Areas:</b> The AI caught \"Diffusion Noise\" (repetitive AI pixel patterns) or melting logic errors in these specific spots.<br>🔵 <b class='text-blue-400'>Blue Areas:</b> The AI considered these areas irrelevant."

            result = {
                "branch_used": "Branch B (General AI Art)",
                "status": "No Face Detected",
                "fake_confidence": fake_prob,
                "real_confidence": real_prob,
                "verdict": "Real Photograph" if real_prob > 50 else "AI-Generated",
                "heatmap_url": f"/static/{heatmap_filename}",
                "explanation": explanation
            }

        os.remove(filepath)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)