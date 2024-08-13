from ultralytics import YOLO

#CARGAR EL MODELO

model = YOLO('Modelos/yolov8l.pt')

model.export(format = 'onnix')