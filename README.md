# BISIVis: Hand Sign Recognition & Data Collection

BISIVis is a Python-based project for collecting, processing, and recognizing hand signs using computer vision and machine learning. It leverages MediaPipe for hand landmark detection and scikit-learn for model training and evaluation.

## Features

- **Data Collection**: Capture hand sign data from a webcam and save it as CSV files for each label.
- **Data Cleaning & Preprocessing**: Remove outliers, handle missing values, and standardize features for robust model training.
- **Model Training**: Train and evaluate machine learning models (Logistic Regression, SVM) to classify hand signs.
- **Model Selection**: Use GridSearchCV to find the best hyperparameters for SVM.
- **Real-Time Recognition**: Predict hand signs from live webcam input using trained models.

## Project Structure

- `collector.py`: Collects hand sign data via webcam and saves it to CSV.
- `builder.ipynb`: Jupyter notebook for data cleaning, visualization, model training, and evaluation.
- `tester.py`: Loads trained models and performs real-time hand sign recognition.
- `data/`: Contains raw and processed CSV files for each hand sign label.
- `model/`: Stores trained model and scaler files (`.pkl`).

## How It Works

1. **Data Collection**: Run `collector.py` to collect hand sign samples for each label. Each sample is a 60-dimensional vector representing hand landmarks.
2. **Data Processing**: Use `builder.ipynb` to clean, visualize, and combine data. Outliers are removed, and features are standardized.
3. **Model Training**: Train models to classify hand signs. The best model and scaler are saved for later use.
4. **Recognition**: Run `tester.py` to recognize hand signs in real-time using your webcam.

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- MediaPipe
- scikit-learn
- pandas, numpy
- joblib

Install dependencies with:

```powershell
pip install opencv-python mediapipe scikit-learn pandas numpy joblib
```

## Usage

- **Collect Data**:  
  Run `collector.py` and follow prompts to collect samples for each hand sign.

- **Build & Train Model**:  
  Open and run `builder.ipynb` to process data and train models.

- **Test Recognition**:  
  Run `tester.py` to start real-time hand sign recognition.

## Notes

- The project uses 21 hand landmarks (excluding the wrist) for each sign.
- Z-coordinates are less reliable due to 2D image limitations; models can be trained with or without Z features.
- Data cleaning is stricter for X and Y, more lenient for Z.

## License

MIT License

---

Feel free to contribute or customize for your own hand sign recognition tasks!
