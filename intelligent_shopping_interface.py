import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkinter.ttk import Progressbar, Style
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

class ShopIAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tienda AI 1.0")
        self.root.geometry("1300x800")

        # Progressbar for Loading
        self.loading_label = tk.Label(root, text="Cargando...", font=('Arial', 16))
        self.loading_label.pack(pady=20)
        self.progress = Progressbar(root, orient=tk.HORIZONTAL, length=400, mode='indeterminate')
        self.progress.pack(pady=10)
        self.progress.start()

        # Inicializar Cámara
        self.initialize()

    def initialize(self):
        # Video Capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se puede abrir la cámara")
            self.root.destroy()
            return

        self.cap.set(3, 1280)
        self.cap.set(4, 700)

        # Cargar Modelos
        self.ObjectModel = YOLO('Modelos/yolov8l.onnx', task='detect')
        self.billModel = YOLO('Modelos/billBank2.onnx', task='detect')

        # Hide Loading Screen
        self.loading_label.pack_forget()
        self.progress.pack_forget()



        # Configuración de estilos de los botones
        style = Style()
        style.configure('TButton',
                        font=('Arial', 12),
                        padding=10,
                        relief='flat',
                        background='lightgray')

        # UI Elements
        self.canvas = tk.Canvas(root, width=1280, height=700, bg='black')
        self.canvas.pack(pady=10)

        # Frame for buttons
        button_frame = tk.Frame(root, bg='lightgray')
        button_frame.pack(pady=10, fill='x', padx=10)

        # Frame for buttons content
        button_frame_content = tk.Frame(button_frame, bg='lightgray')
        button_frame_content.pack(expand=True)

        # Camera Control Buttons
        self.start_button = ttk.Button(button_frame_content, text="Iniciar Detección", command=self.start_detection)
        self.start_button.grid(row=0, column=0, padx=10, pady=5)

        self.stop_button = ttk.Button(button_frame_content, text="Detener Cámara", command=self.stop_camera)
        self.stop_button.grid(row=0, column=1, padx=10, pady=5)

        self.reset_button = ttk.Button(button_frame_content, text="Reiniciar", command=self.reset)
        self.reset_button.grid(row=0, column=2, padx=10, pady=5)

        # Center buttons in button_frame_content
        button_frame_content.grid_columnconfigure(0, weight=1)
        button_frame_content.grid_columnconfigure(1, weight=1)
        button_frame_content.grid_columnconfigure(2, weight=1)

        # Payment Processing Button
        self.process_payment_button = ttk.Button(root, text="Procesar Pago", command=self.process_payment)
        self.process_payment_button.pack(pady=20)

        # Status Label
        self.status_label = tk.Label(root, text="Estado: Listo", font=('Arial', 14), bg='lightgray')
        self.status_label.pack(pady=10)

        # State Variables
        self.detection_active = False
        self.total_balance = 0
        self.accumulative_price = 0

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
            else:
                print("Error al capturar el frame")

        self.root.after(10, self.update_frame)

    def draw_results(self, frame, results_objects, results_bills):
        # Draw bounding boxes for objects
        for result in results_objects:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    if hasattr(box, 'xyxy') and box.xyxy is not None:
                        xyxy = box.xyxy.cpu().numpy() if hasattr(box.xyxy, 'cpu') else box.xyxy.numpy()
                        xyxy = xyxy[0] if len(xyxy.shape) > 1 else xyxy
                        x1, y1, x2, y2 = map(int, xyxy)
                        label = self.ObjectModel.names[int(box.cls)]
                        frame = self.draw_bounding_box(frame, x1, y1, x2, y2, label)

        # Draw bounding boxes for bills
        for result in results_bills:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    if hasattr(box, 'xyxy') and box.xyxy is not None:
                        xyxy = box.xyxy.cpu().numpy() if hasattr(box.xyxy, 'cpu') else box.xyxy.numpy()
                        xyxy = xyxy[0] if len(xyxy.shape) > 1 else xyxy
                        x1, y1, x2, y2 = map(int, xyxy)
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
            messagebox.showinfo("Pago", f"Falta cancelar {self.accumulative_price - self.total_balance}$")
        elif self.accumulative_price < self.total_balance:
            messagebox.showinfo("Pago", f"Su cambio es de: {self.total_balance - self.accumulative_price}$")
        else:
            messagebox.showinfo("Pago", "¡Gracias por su compra!")

    def start_detection(self):
        self.detection_active = True
        self.status_label.config(text="Estado: Detección Activa")
        self.update_frame()

    def stop_camera(self):
        self.detection_active = False
        self.cap.release()
        self.canvas.delete("all")
        self.status_label.config(text="Estado: Cámara Detenida")

    def reset(self):
        self.detection_active = False
        self.total_balance = 0
        self.accumulative_price = 0
        self.canvas.delete("all")
        if self.cap.isOpened():
            self.cap.release()
        # Reinitialize the video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se puede abrir la cámara después del reinicio")
        self.status_label.config(text="Estado: Listo")


if __name__ == "__main__":
    root = tk.Tk()
    app = ShopIAApp(root)
    root.mainloop()
