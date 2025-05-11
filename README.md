plant-disease-detection
Plant Disease Classification using CNN and Transfer Learning

Objectives:

1.Classify plant leaf images into specific disease categories
2.Use real-world image dataset for training (Potato: Early Blight, Late Blight, Healthy)
3.Build a lightweight CNN model using MobileNetV2
4.Evaluate performance using Accuracy and F1 Score
5.Predict disease from user-uploaded leaf images
6.Enable easy testing and future deployment

Features:
1.Deep learning classification with Keras/TensorFlow
2.Pretrained MobileNetV2 base model
3.Data augmentation for improved generalization
4.Image upload + prediction in real-time
5.Evaluation metrics: Accuracy, F1 score, classification report
6.Simple, Colab-compatible workflow

Programming Language & Tools:
Language: Python

Libraries: TensorFlow, Keras, NumPy, sklearn, Matplotlib, PIL

Dataset: Plant Disease Dataset - Akshit Gupta (Kaggle)

Evaluation:
Accuracy: ~42%

F1 Score (macro): ~32%

Predicted correctly: Potato___Early_blight from uploaded leaf image

Works best on disease classes with balanced data

Future Project Perspective:
1.Advanced CNN: Use ResNet, EfficientNet for better performance
2.Mobile App: Capture & classify disease using camera (Android/iOS)
3.Offline Mode: Make predictions without internet using TFLite
4.Dashboard: Visual analytics for farmers & agri-scientists
5.Multicrop Extension: Support more crops (tomato, maize, etc.)

