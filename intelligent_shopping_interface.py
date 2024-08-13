import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

class ShopIAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ShopIA")

        # Video Capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        # Load Models
        self.ObjectModel = YOLO('Modelos/yolov8l.onnx', task='detect')
        self.billModel = YOLO('Modelos/billBank2.onnx', task='detect')

        # UI Elements
        self.canvas = tk.Canvas(root, width=1280, height=720)
        self.canvas.pack()

        # Buttons
        self.start_button = tk.Button(root, text="Start Detection", command=self.start_detection)
        self.start_button.pack()

        self.process_payment_button = tk.Button(root, text="Process Payment", command=self.process_payment)
        self.process_payment_button.pack()

        self.stop_button = tk.Button(root, text="Stop Camera", command=self.stop_camera)
        self.stop_button.pack()

        self.reset_button = tk.Button(root, text="Reset", command=self.reset)
        self.reset_button.pack()

        # State Variables
        self.detection_active = False
        self.total_balance = 0
        self.accumulative_price = 0
        self.update_frame()

    def update_frame(self):
        if self.detection_active:
            ret, frame = self.cap.read()
            if ret:
                # Perform object and bill detection
                results_objects = self.ObjectModel(frame)
                results_bills = self.billModel(frame)

                # Draw detection results on the frame
                self.draw_results(frame, results_objects, results_bills)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.canvas.imgtk = imgtk

        self.root.after(10, self.update_frame)

    def draw_results(self, frame, results_objects, results_bills):
        # Draw bounding boxes for objects
        for result in results_objects:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    if hasattr(box, 'xyxy') and box.xyxy is not None:
                        xyxy = box.xyxy.cpu().numpy() if hasattr(box.xyxy, 'cpu') else box.xyxy.numpy()  # Ensure it's a numpy array
                        xyxy = xyxy[0] if len(xyxy.shape) > 1 else xyxy  # Handle single box case
                        x1, y1, x2, y2 = map(int, xyxy)  # Convert to integer coordinates
                        label = self.ObjectModel.names[int(box.cls)]
                        frame = self.draw_bounding_box(frame, x1, y1, x2, y2, label)

        # Draw bounding boxes for bills
        for result in results_bills:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    if hasattr(box, 'xyxy') and box.xyxy is not None:
                        xyxy = box.xyxy.cpu().numpy() if hasattr(box.xyxy, 'cpu') else box.xyxy.numpy()  # Ensure it's a numpy array
                        xyxy = xyxy[0] if len(xyxy.shape) > 1 else xyxy  # Handle single box case
                        x1, y1, x2, y2 = map(int, xyxy)  # Convert to integer coordinates
                        label = self.billModel.names[int(box.cls)]
                        frame = self.draw_bounding_box(frame, x1, y1, x2, y2, label)

    def draw_bounding_box(self, frame, x1, y1, x2, y2, label):
        color = (0, 255, 0)  # Green color for bounding boxes
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def process_payment(self):
        # Dummy payment processing
        if self.accumulative_price > self.total_balance:
            messagebox.showinfo("Payment", f"Falta cancelar {self.accumulative_price - self.total_balance}$")
        elif self.accumulative_price < self.total_balance:
            messagebox.showinfo("Payment", f"Su cambio es de: {self.total_balance - self.accumulative_price}$")
        else:
            messagebox.showinfo("Payment", "Gracias por su compra!")

    def start_detection(self):
        self.detection_active = True

    def stop_camera(self):
        self.detection_active = False
        self.cap.release()
        self.canvas.delete("all")

    def reset(self):
        self.detection_active = False
        self.total_balance = 0
        self.accumulative_price = 0
        self.cap.release()
        self.canvas.delete("all")

if __name__ == "__main__":
    root = tk.Tk()
    app = ShopIAApp(root)
    root.mainloop()
