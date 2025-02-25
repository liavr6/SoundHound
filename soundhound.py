import sys
import librosa
import numpy as np
import torch
import librosa.display
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import QThread, pyqtSignal
from speechbrain.pretrained import SpeakerRecognition

class SpeakerVerificationThread(QThread):
    result_ready = pyqtSignal(str, bool)

    def __init__(self, sr_model, reference_audio, test_audio):
        super().__init__()
        self.sr_model = sr_model
        self.reference_audio = reference_audio
        self.test_audio = test_audio

    def run(self):
        similarity = self.sr_model.verify_batch(
            torch.tensor(self.reference_audio).unsqueeze(0),
            torch.tensor(self.test_audio).unsqueeze(0)
        )
        score = similarity[0].item() if isinstance(similarity, tuple) else similarity.item()
        is_match = score >= 0.5
        result_text = "✅ Same Speaker" if is_match else "❌ Different Speaker"
        self.result_ready.emit(result_text, is_match)

class VoiceVerificationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.sr_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                         savedir="tmp_model")
        self.reference_audio = None

    def initUI(self):
        self.setWindowTitle("Speaker Verification with Frequency Analysis")
        self.setGeometry(100, 100, 700, 500)

        layout = QVBoxLayout()

        self.label = QLabel("Load a reference audio file")
        self.label.setFont(QFont("Arial", 14))
        layout.addWidget(self.label)

        self.loadButton = QPushButton("Load Reference Audio")
        self.loadButton.clicked.connect(self.load_reference_audio)
        layout.addWidget(self.loadButton)

        self.compareButton = QPushButton("Compare with New Audio")
        self.compareButton.clicked.connect(self.compare_audio)
        self.compareButton.setEnabled(False)
        layout.addWidget(self.compareButton)

        self.freqPlotLabel = QLabel()
        layout.addWidget(self.freqPlotLabel)

        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

    def load_reference_audio(self):
        self.reference_audio = self.load_audio("Select Reference Audio File")
        if self.reference_audio is not None:
            self.compareButton.setEnabled(True)

    def compare_audio(self):
        test_audio = self.load_audio("Select Audio File to Compare")
        if test_audio is not None:
            self.compareButton.setEnabled(False)
            self.label.setText("Processing... Please wait.")

            self.worker_thread = SpeakerVerificationThread(self.sr_model, self.reference_audio, test_audio)
            self.worker_thread.result_ready.connect(self.update_result)
            self.worker_thread.start()

            self.visualize_frequency_profiles(self.reference_audio, test_audio)

    def update_result(self, result_text, is_match):
        self.label.setText(result_text)
        self.compareButton.setEnabled(True)

    def load_audio(self, dialog_title):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, dialog_title, "", "Audio/Video Files (*.wav *.mp3 *.mp4)", options=options)
        if not filePath:
            return None

        if filePath.endswith(".mp4"):
            audio_path = "temp_audio.wav"
            clip = VideoFileClip(filePath)
            clip.audio.write_audiofile(audio_path, codec="pcm_s16le")
            filePath = audio_path  

        audio, _ = librosa.load(filePath, sr=16000, mono=True)
        return audio

    def visualize_frequency_profiles(self, reference_audio, test_audio):
        # Compute the frequency spectrum for both speakers
        freqs_ref, power_ref = self.compute_frequency_profile(reference_audio)
        freqs_test, power_test = self.compute_frequency_profile(test_audio)

        # Compute the center of mass of the frequency profile
        center_ref = self.compute_center_of_mass(freqs_ref, power_ref)
        center_test = self.compute_center_of_mass(freqs_test, power_test)

        # Determine the limits for zooming (± 500 Hz from the center of mass)
        zoom_margin = 500  # Define how much to zoom around the center
        min_freq = min(center_ref, center_test) - zoom_margin
        max_freq = max(center_ref, center_test) + zoom_margin

        # Plot both frequency profiles
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(freqs_ref, power_ref, label="Reference Speaker", color="blue")
        ax.plot(freqs_test, power_test, label="Test Speaker", color="red", linestyle="dashed")

        ax.set_title("Frequency Profile Comparison")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Normalized Power")

        # Set the x-axis limits based on the center of mass and zoom
        ax.set_xlim(min_freq, max_freq)

        # Dynamically set the y-axis limits to focus on the peak of the power spectrum
        ax.set_ylim(0, max(np.max(power_ref), np.max(power_test)) * 1.1)  # Adding a 10% margin on the y-axis

        ax.legend()
        ax.grid()

        # Save the plot and display it
        plt.savefig("frequency_profile.png")
        plt.close(fig)

        pixmap = QPixmap("frequency_profile.png")
        self.freqPlotLabel.setPixmap(pixmap)

    def compute_center_of_mass(self, freqs, power):
        # Calculate the center of mass (weighted average of frequency)
        total_power = np.sum(power)
        center_of_mass = np.sum(freqs * power) / total_power
        return center_of_mass



    def compute_frequency_profile(self, audio):
        # Compute FFT
        fft_result = np.abs(np.fft.rfft(audio)) ** 2  # Power spectrum
        freqs = np.fft.rfftfreq(len(audio), d=1/16000)  # Frequency bins

        # Normalize power to range [0, 1]
        power_normalized = fft_result / np.max(fft_result)

        return freqs, power_normalized

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = VoiceVerificationApp()
    mainWin.show()
    sys.exit(app.exec_())
