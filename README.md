# Distracted Driver Detection Using CNN

This project aims to detect and classify distracted driving behaviors using dashboard camera images. A deep learning model based on **Convolutional Neural Networks (CNNs)** was built to identify whether a driver is engaged in unsafe activities such as texting, talking on the phone, eating, or is safely driving.

## Project Objective

To automate the detection of distracted driver behavior by classifying images into predefined categories using deep learning, helping improve road safety through computer vision techniques.

## Classes Detected

- Safe Driving  
- Texting (Right / Left Hand)  
- Talking on the Phone (Right / Left Hand)  
- Operating the Radio  
- Drinking  
- Reaching Behind  
- Hair and Makeup  
- Talking to Passenger  

## Technologies Used

- **Programming Language**: Python  
- **Deep Learning Frameworks**: TensorFlow, Keras  
- **Image Processing**: OpenCV  
- **Data Handling**: NumPy, Pandas  
- **Visualization**: Matplotlib, Seaborn  
- **Model Evaluation**: Scikit-learn  
- **Data Augmentation**: ImageDataGenerator  

## Model Architecture

- Custom **Convolutional Neural Network (CNN)** with multiple Conv2D and MaxPooling layers  
- Activation functions: ReLU, Softmax  
- Optimizer: Adam  
- Loss function: Categorical Crossentropy  
- Evaluation metrics: Accuracy, Precision, Recall, F1-score

## Workflow

1. **Data Loading**: Images categorized into folders by class label  
2. **Data Preprocessing**: Resizing, normalization, augmentation (flip, rotation, zoom)  
3. **Model Building**: Defined and compiled a CNN using Keras  
4. **Training**: Model trained on labeled dataset using train-validation split  
5. **Evaluation**: Analyzed confusion matrix, classification report, and loss/accuracy curves  
6. **Prediction**: Model tested on unseen images to validate generalization

## Performance Metrics

- Accuracy: (Add final accuracy if known, e.g., 93%)  
- Loss: Tracked across epochs  
- Confusion Matrix: Used to evaluate misclassifications  
- F1-Score, Precision, and Recall: Computed for each class

## Folder Structure
├── dataset/
│ ├── c0/ # Safe driving
│ ├── c1/ # Texting - right
│ ├── c2/ # Talking on the phone - right
│ └── ...
├── model/
│ ├── model.h5
│ └── history_plot.png
├── driver_detection.ipynb
├── README.md


## Future Enhancements

- Integrate real-time webcam prediction support using OpenCV  
- Convert model for mobile deployment using TensorFlow Lite  
- Implement transfer learning using pretrained models like ResNet or MobileNet  
- Add a Flask web interface for interactive demo

## License

This project is intended for educational and research purposes only. Modify and use as needed.
