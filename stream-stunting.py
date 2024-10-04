import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Buat tampilan web menggunakan Streamlit
st.title('Aplikasi Klasifikasi Status Stunting')

# Upload dataset
uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataframe:")
    st.write(df)

    # Data preprocessing
    df['BB_Lahir'].replace(0, np.nan, inplace=True)
    df['TB_Lahir'].replace(0, np.nan, inplace=True)
    df = df.dropna()
    df = df.drop(columns=['Tanggal_Pengukuran'])

    # Label Encoding
    encode = LabelEncoder()
    df['JK'] = encode.fit_transform(df['JK'].values)
    df['TB_U'] = encode.fit_transform(df['TB_U'].values)
    df['Status'] = encode.fit_transform(df['Status'].values)

    # Menentukan X dan y
    st.header('Data Selection')
    X = df[['JK', 'Umur', 'Berat','Tinggi', 'BB_Lahir', 'TB_Lahir', 'ZS_TB_U']]
    st.write('Features (X):')
    st.write(X)
    
    y = df['Status']
    st.write('Target (y):')
    st.write(y)

    # Skalakan fitur ke rentang [0, 1] menggunakan MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Terapkan SMOTE pada seluruh dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # One-Hot Encoding untuk label
    y_resampled_encoded = to_categorical(y_resampled, num_classes=3)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled_encoded, test_size=0.2, random_state=42)
    st.write('X_train:')
    st.write(X_train)
    
    st.write('X_test:')
    st.write(X_test)

    # Menambahkan dimensi waktu (1) ke data pelatihan dan pengujian
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Tambahkan input parameter
    st.header('Pilih Parameter Training')
    available_neurons = [16, 32, 64, 128, 256, 512, 1024]
    available_epochs = [10, 20, 50, 100, 200, 500]
    available_batch_sizes = [32, 64, 128, 256, 512, 1024]
    neurons = st.select_slider('Jumlah Neuron', options=available_neurons)
    epochs = st.select_slider('Epoch', options=available_epochs)
    batch_size = st.select_slider('Batch Size', options=available_batch_sizes)

    # Tambahkan tombol untuk melatih model
    if st.button('Latih Model'):
        # Membangun Model LSTM
        model = Sequential()
        model.add(LSTM(neurons, input_shape=(X_train.shape[1], 1), return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(neurons, return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))

        # Compile Model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Latih model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

        # Evaluasi model
        loss, accuracy = model.evaluate(X_test, y_test)
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        st.subheader('Evaluasi Model')
        st.write(f"Loss: {loss:.4f}")
        st.write(f"Accuracy: {accuracy:.4f}")
        
        # Plot accuracy dan loss
        st.subheader('Grafik Akurasi dan Loss')
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        ax[0].plot(history.history['accuracy'], label='Train Accuracy')
        ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend()
        ax[0].set_title('Accuracy')

        ax[1].plot(history.history['loss'], label='Train Loss')
        ax[1].plot(history.history['val_loss'], label='Validation Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        ax[1].legend()
        ax[1].set_title('Loss')

        st.pyplot(fig)

        # Tampilkan Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        # Tampilkan Confusion Matrix dalam bentuk tabel
        st.subheader('Actual v LSTM')
        cm_df = pd.DataFrame(cm, index=['Actual Normal', 'Actual Severely Stunting', 'Actual Stunting'], columns=['LSTM Normal', 'LSTM Severely Stunting', 'LSTM Stunting'])
        st.write(cm_df)

        # Tampilkan tabel data aktual dan hasil klasifikasi LSTM
        st.subheader('Tabel Data Aktual dan Hasil Klasifikasi LSTM')
        results_df = pd.DataFrame({'Actual': y_true, 'LSTM': y_pred_classes})
        st.write(results_df)