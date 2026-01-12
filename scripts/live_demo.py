import cv2
import torch
import sys
import os
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor

# Add root to path so we can import 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.factory import AdapterFactory

def live_inference():
    # 1. Load the Phase 2 Engine (Untrained Adapter)
    print("Loading Model...")
    # Use the master template we created earlier
    if not os.path.exists("configs/master_template.yaml"):
        print("❌ Error: Config file not found.")
        return

    model, cfg = AdapterFactory.create_model("configs/master_template.yaml")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # 2. Setup Preprocessor (Must match ResNet-50)
    processor = AutoImageProcessor.from_pretrained(cfg['model']['base_model'])

    # 3. Open Webcam (0 is usually the default cam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        print("   (Note: WSL usually cannot see the webcam directly without USBIP setup.)")
        return

    print("✅ Camera Active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 4. Preprocess Frame for ResNet
        # OpenCV uses BGR, PIL uses RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        inputs = processor(pil_image, return_tensors="pt").to(device)

        # 5. Run Inference (The Engine at work!)
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # Get prediction (It will be random since we haven't trained!)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        conf, pred_class = torch.max(probs, dim=-1)

        # 6. Draw on Screen
        label = f"Class: {pred_class.item()} | Conf: {conf.item():.2f}"
        
        # Red box if low confidence, Green if high
        color = (0, 255, 0) if conf.item() > 0.5 else (0, 0, 255)
        
        cv2.putText(frame, "PHASE 2 ENGINE ACTIVE", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, label, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Uncertainty Gated LoRA - Live Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_inference()
