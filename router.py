import cv2
from mtcnn import MTCNN

class PipelineRouter:
    def __init__(self):
        print("Loading MTCNN Router...")
        self.detector = MTCNN()

    def route_image(self, image_path):
        # Dynamic path loading
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(img_rgb)
        
        if len(faces) > 0:
            print(f"Router Status: Found {len(faces)} face(s). Activating Branch A & B.")
            cropped_faces = []
            
            for i, face in enumerate(faces):
                x, y, w, h = face['box']
                x, y = abs(x), abs(y)
                
                # 30% padding
                pad_x = int(w * 0.30)
                pad_y = int(h * 0.30)

                start_y = max(0, y - pad_y)
                end_y = min(img_rgb.shape[0], y + h + pad_y)
                start_x = max(0, x - pad_x)
                end_x = min(img_rgb.shape[1], x + w + pad_x)
                
                cropped_face = img_rgb[start_y:end_y, start_x:end_x]
                cropped_faces.append(cropped_face)

            return {
                "route": "BOTH",
                "full_image": img_rgb,      
                "cropped_faces": cropped_faces 
            }
            
        else:
            print("Router Status: No faces detected. Activating Branch B only.")
            return {
                "route": "BRANCH_B",
                "full_image": img_rgb,      
                "cropped_faces": None
            }