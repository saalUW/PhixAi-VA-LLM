import os
import soundfile as sf

#Goal: Trim end of audio clip
#Reason: Save time/money on API calls for silent audio if the clip is ended early

def trim_audio(file_path):
    # Load audio file
    audio, sample_rate = sf.read(file_path)

    # Find the index of the last non-silent sample
    last_non_silent_index = len(audio) - 1
    # Can adjust in the future to better optimize and ensure nothing is being lost
    while last_non_silent_index >= 0 and abs(audio[last_non_silent_index]) < 0.01:
        last_non_silent_index -= 1

    # Trim audio by removing trailing silence
    trimmed_audio = audio[:last_non_silent_index + 1]
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Save trimmed audio as a new file
    trimmed_file_path = os.path.join("records", f"{file_name}.wav")
    sf.write(trimmed_file_path, trimmed_audio, sample_rate)
    print(f"Trimmed audio saved as {trimmed_file_path}.")

# Trim all .wav files in the "records" folder
folder_path = "records"
for file_name in os.listdir(folder_path):
    if file_name.endswith(".wav"):
        file_path = os.path.join(folder_path, file_name)
        trim_audio(file_path)
