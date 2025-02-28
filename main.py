Creating a comprehensive Python program for a project like "Smart-Health-Monitor" requires integration with IoT devices, data processing, and applying machine learning models. I'll provide a simplified version that outlines the major components: data gathering, processing, modeling, and real-time monitoring. For the sake of this example, we'll simulate IoT device data and use a basic machine learning model to predict health anomalies.

To execute this code, additional real-time data from IoT devices and specific health anomaly models will be necessary, which is beyond this simple illustration. Libraries like `scikit-learn` for machine learning and `pandas` for data handling will be used.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

# Simulate IoT data
def simulate_iot_data():
    # Simulate data from IoT devices like heart rate, temperature, etc.
    # For simplicity, let's simulate heart rate (bpm) and temperature (Celsius)
    heart_rate = np.random.randint(50, 130)
    temperature = np.random.uniform(36.5, 39.5)
    return heart_rate, temperature

# Define a simple function for generating a dataset
def generate_dataset(n_samples=1000):
    # Simulate dataset with heart_rate, temperature and a label for anomaly
    data = {
        "heart_rate": np.random.randint(50, 130, n_samples),
        "temperature": np.random.uniform(36.5, 39.5, n_samples),
        "anomaly": np.random.choice([0, 1], n_samples, p=[0.95, 0.05]) # 5% chance of anomaly
    }
    return pd.DataFrame(data)

# Prepare the dataset
def prepare_data():
    dataset = generate_dataset()
    
    # Splitting the dataset
    X = dataset[['heart_rate', 'temperature']]
    y = dataset['anomaly']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Main function for real-time monitoring
def real_time_monitoring(model):
    try:
        while True:
            heart_rate, temperature = simulate_iot_data()
            data_point = np.array([[heart_rate, temperature]])
            
            # Predicting anomaly
            prediction = model.predict(data_point)
            
            if prediction[0] == 1:
                print("Anomaly Detected! Heart Rate:", heart_rate, "Temperature:", temperature)
            else:
                print("Normal Health Metrics. Heart Rate:", heart_rate, "Temperature:", temperature)
            
            # Sleep for 5 seconds to simulate real-time monitoring
            time.sleep(5)

    except KeyboardInterrupt:
        print("Stopping real-time monitoring.")
    except Exception as e:
        print("An error occurred:", str(e))

# Error handling for modeling
def main():
    try:
        X_train, X_test, y_train, y_test = prepare_data()
        model = train_model(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

        # Start real-time monitoring
        real_time_monitoring(model)
    
    except Exception as e:
        print("An error occurred in the main function:", str(e))

# Run the program
if __name__ == "__main__":
    main()
```

### Description:
- **simulate_iot_data**: Simulates incoming data from an IoT device for heart rate and temperature.
- **generate_dataset**: Generates a random dataset to simulate training data.
- **prepare_data**: Prepares and splits the dataset into training and testing sets.
- **train_model**: Trains a RandomForestClassifier on the training data.
- **real_time_monitoring**: A continuous loop that simulates reading from IoT devices and uses the model to predict anomalies. It runs until interrupted, typically with `Ctrl + C`.
- **Error Handling**: Basic error handling provided to manage exceptions during execution. 

In a real application, you would replace the simulation parts with real data sources and potentially use more complex models and data preprocessing steps suitable for your specific use case.