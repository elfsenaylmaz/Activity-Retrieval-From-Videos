import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import torch
from ViVit import VideoVisionTransformer
import numpy as np
import torch.nn.functional as F
from tkinter import font

class NormalizeVideo:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

class VideoPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Activity Retrieval From Videos")
        self.root.geometry("400x300")
        self.video_path = None
        self.model_path = None

        custom_font = font.Font(family="Courier", size=12, weight="bold")
        self.text_label = tk.Label(root, text="Prof. Dr. Mine Elif Karslıgil\nEngin Memiş - Elif Sena Yılmaz\n   19011040 - 20011040", font=custom_font, anchor="center", justify="left")
        self.text_label.pack(fill="both", padx=10, pady=10)

        # Create buttons
        self.upload_button = tk.Button(root, text="Bilgisayardan Video Yükle", command=self.upload_video)
        self.upload_button.pack(pady=10)

        self.model_button = tk.Button(root, text="Model Seç", command=self.select_model)
        self.model_button.pack(pady=10)

        self.output_button = tk.Button(root, text="Output Oluştur", command=self.create_output)
        self.output_button.pack(pady=10)

        self.camera_button = tk.Button(root, text="Canlı Kamera", command=self.camera)
        self.camera_button.pack(pady=10)

    def upload_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if self.video_path:
            messagebox.showinfo("Selected Video", f"Selected video: {self.video_path}")
        else:
            messagebox.showwarning("No Video Selected", "Please select a video file.")

    def select_model(self):
        self.model_path = filedialog.askopenfilename(filetypes=[("PyTorch Model files", "*.pth")])
        if self.model_path:
            messagebox.showinfo("Selected Model", f"Selected model: {self.model_path}")
        else:
            messagebox.showwarning("No Model Selected", "Please select a .pth model file.")

    def camera(self):
        if not self.model_path:
            messagebox.showwarning("Missing Input", "Please select a model.")
            return
        
        with open('Classes.txt', 'r', encoding='utf-8') as file:
            classesList = file.readlines()

        classesList = [name.split("\n")[0] for name in classesList]
        
        # Normalizasyon işlemi için ortalama ve standart sapma değerleri
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = NormalizeVideo(mean, std)

        # Load the model
        model = VideoVisionTransformer().cuda()
        model.load_state_dict(torch.load(self.model_path))
        model.eval()


        cap = cv2.VideoCapture(0)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        outFrames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            outFrame = frame
            outFrames.append(outFrame)
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            frames.append(frame)

            if len(frames) >= 64:
                frames = frames[-64:]

                framesTensor = np.array(frames).astype(np.float32)  # (T, H, W, C)
                framesTensor = torch.tensor(framesTensor).permute(3, 0, 1, 2)  # (C, T, H, W)
                framesTensor= normalize(framesTensor / 255.0)

                framesTensor = framesTensor.unsqueeze(0) 

                framesTensor = framesTensor.cuda()
                with torch.no_grad():
                    output = model(framesTensor)
                

                probabilities = F.softmax(output, dim=1)
                values, indices = torch.topk(probabilities, 5, axis = 1)

                predictionString = f"3. {classesList[indices[0][2]]}: {values[0][2]:.2f}"
                newFrame = self.add_prediction_to_frame(outFrame = outFrame, prediction=predictionString, outFrameHeight=height, outFrameWidth=width)

                predictionString = f"2. {classesList[indices[0][1]]}: {values[0][1]:.2f}"
                newFrame = self.add_prediction_to_frame(outFrame = newFrame, prediction=predictionString, outFrameHeight=height, outFrameWidth=width)

                predictionString = f"1. {classesList[indices[0][0]]}: {values[0][0]:.2f}"
                newFrame = self.add_prediction_to_frame(outFrame = newFrame, prediction=predictionString, outFrameHeight=height, outFrameWidth=width)
                    

                #cv2.imshow('Action Recognition', frame)
                cv2.imshow('Action Recognition', newFrame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()        
                

    def create_output(self):
        if not self.video_path or not self.model_path:
            messagebox.showwarning("Missing Input", "Please select both a video file and a model.")
            return
    
        with open('Classes.txt', 'r', encoding='utf-8') as file:
            classesList = file.readlines()

        classesList = [name.split("\n")[0] for name in classesList]
        
        # Normalizasyon işlemi için ortalama ve standart sapma değerleri
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = NormalizeVideo(mean, std)

        # Load the model
        model = VideoVisionTransformer().cuda()
        model.load_state_dict(torch.load(self.model_path))
        model.eval()


        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        outFrames = []

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_idx = total_frames // 3
        frame_idx -= 64 // 2

        if frame_idx < 0:
            frame_idx = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            outFrameHeight, outFrameWidth = height , width
            if outFrameWidth < 400:
                ratio = outFrameHeight / outFrameWidth
                outFrameWidth = 400
                outFrameHeight *= ratio
                outFrameHeight = int(outFrameHeight)

            outFrame = cv2.resize(frame, (outFrameWidth, outFrameHeight))
            outFrames.append(outFrame)
            frame = cv2.resize(frame, (224, 224))
            #out.write(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frames.append(frame)
            if len(frames) >= 64:
                break

        cap.release()
        #out.release()

        if len(frames) == 0:
            # If no frames are captured, create a dummy frame filled with zeros
            frames = [np.zeros((224, 224, 3), dtype=np.float32) for _ in range(64)]
            outFrames = [np.zeros((224, 224, 3), dtype=np.float32) for _ in range(64)]
        elif len(frames) < 64:
            # Pad with the last frame if video is shorter than self.num_frames
            while len(frames) < 64:
                frames.append(frames[-1])
                outFrames.append(frames[-1])
        elif len(frames) > 64:
            # Truncate frames if video is longer than self.num_frames
            frames = frames[:64]
            outFrames = outFrames[:64]

        framesTensor = np.array(frames).astype(np.float32)  # (T, H, W, C)
        framesTensor = torch.tensor(framesTensor).permute(3, 0, 1, 2)  # (C, T, H, W)
        framesTensor = normalize(framesTensor / 255.0)

        framesTensor = framesTensor.unsqueeze(0)
        framesTensor = framesTensor.cuda()

        with torch.no_grad():
            prediction = model(framesTensor)

        probabilities = F.softmax(prediction, dim=1)
        values, indices = torch.topk(probabilities, 5, axis = 1)

        combinedFrames = []
        predictionString = f"3. {classesList[indices[0][2]]}: {values[0][2]:.2f}"
        for outFrame in outFrames:
            newFrame = self.add_prediction_to_frame(outFrame = outFrame, prediction=predictionString, outFrameHeight=outFrameHeight, outFrameWidth=outFrameWidth)
            combinedFrames.append(newFrame)

        combinedFrames2 = []
        predictionString = f"2. {classesList[indices[0][1]]}: {values[0][1]:.2f}"
        for newFrame in combinedFrames:
            newFrame = self.add_prediction_to_frame(outFrame = newFrame, prediction=predictionString, outFrameHeight=outFrameHeight, outFrameWidth=outFrameWidth)
            combinedFrames2.append(newFrame)

        combinedFrames3 = []
        predictionString = f"1. {classesList[indices[0][0]]}: {values[0][0]:.2f}"
        for newFrame in combinedFrames2:
            newFrame = self.add_prediction_to_frame(outFrame = newFrame, prediction=predictionString, outFrameHeight=outFrameHeight, outFrameWidth=outFrameWidth)
            combinedFrames3.append(newFrame)

        for value, indice in zip(values[0], indices[0]):
            print(f"{classesList[indice]}: {value * 100}")

        i = 0
        while True:
            outFrame = combinedFrames3[i % (len(combinedFrames3))]
            i += 1
            cv2.imshow('Video', outFrame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()

    def preprocess_frame(self, frame):
        # Implement frame preprocessing here (e.g., resizing, normalization)
        # This is just a placeholder
        input_tensor = torch.tensor(frame).unsqueeze(0).float()
        return input_tensor
    
    def add_prediction_to_frame(self, outFrame, prediction, outFrameHeight, outFrameWidth, height=20, color=(255, 255, 255)):
        # Frame'in genişliğini ve yüksekliğini al
        frame_height, frame_width = outFrameHeight, outFrameWidth

        # Beyaz alan yarat (height: beyaz alanın yüksekliği)
        white_area = np.full((height, frame_width, 3), color, dtype=np.uint8)

        # Frame'in boyutunu beyaz alanın genişliğine uyacak şekilde yeniden boyutlandır
        outFrame = cv2.resize(outFrame, (frame_width, frame_height))

        # Frame'i beyaz alanla birleştir
        combined_frame = np.vstack((white_area, outFrame))

        # Prediction metnini beyaz alana yazdır

        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(str(prediction), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = (frame_width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2

        cv2.putText(combined_frame, str(prediction), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        #cv2.putText(combined_frame, str(prediction), (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        return combined_frame

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPredictorApp(root)
    root.mainloop()
