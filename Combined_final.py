import sounddevice as sd
import soundfile as sf
import re
import os
from Audio_trim import *
import glob
import warnings
import csv
from tqdm import tqdm
from transformers import pipeline
import torch
from transformers import pipeline
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import time



### Environment Setup ###

warnings.filterwarnings("ignore")

device = 0 if torch.cuda.is_available() else -1
print("CUDA availability:", torch.cuda.is_available())

#Used for Classify function
class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list
        
    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]
    
### Record Audio ###
def record_audio():
    sample_rate = 44100 
    #sample_rate = 16000 
    print("Countdown:")
    for i in range(3, 0, -1):
        print(f"{i}")
        time.sleep(1)
    print("Start!")
    
    audio = sd.rec(int(sample_rate * 60), samplerate=sample_rate,
                   channels=1, blocking=False)
    
    input("Press Enter to stop.")
    sd.stop()
    
    file_name = input("Enter Unique_ID: ")
    
    # Create "records" directory for the audio clips to be saved in
    directory = "records"
    trans_directory = "trans_records"
    os.makedirs(directory, exist_ok=True)
    os.makedirs(trans_directory, exist_ok=True)

    valid_groups = ["Adult", "Child", "Neonate"]

    while True:
        try:
            mod = input("What Group? (Adult, Child, Neonate): ")
            if mod not in valid_groups:
                raise ValueError("Invalid group. Please enter a valid group.")
            break
        except ValueError as e:
            print(e)

    file_name = file_name + "_" + mod

    # Check if recording is in English - could add translate feature
    while True:
        try:
            is_english = input("Is the recording in English? (Yes/No): ")
            if is_english.lower() not in ["yes", "no"]:
                raise ValueError("Invalid input. Please enter either Yes or No.")
            break
        except ValueError as e:
            print(e)

    if is_english.lower() == "yes":
        file_path = os.path.join(directory, f"{file_name}.wav")
    else:
        file_path = os.path.join(trans_directory, f"{file_name}.wav")

    sf.write(file_path, audio, sample_rate)
    print(f"Audio saved as {file_path}.")

    trim_audio(file_path)
    return file_path

### Transcribe ###
def transcribe_files():

    file_list = glob.glob("records/*.wav")
    transcriptions = []
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2", device=device)

    for file_path in tqdm(file_list, desc="Transcribing files"):
        # Extract the module name from the file_name
        file_name = os.path.basename(file_path)
        module = ""
        if "Adult" in file_name:
            module = "Adult"
        elif "Child" in file_name:
            module = "Child"
        elif "Neonate" in file_name:
            module = "Neonate"

        p = pipe(file_path)
        text = p["text"]
        
        id_match = re.search(r"\d+", file_name)
        if id_match:
            id_name = id_match.group()
        else:
            id_name = "Unknown"

        transcriptions.append([id_name, module, text, "unknown"])

    column_names = ["ID", "module", "open_response", "LLM_CoD"]

    csv_file_path = "transcript.csv"
    with open(csv_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(column_names)  # Write the column names as the first row
        writer.writerows(transcriptions)

    print(f"Transcriptions saved as {csv_file_path}.")

### Translate ###
def translate_files():
    folder_path = "trans_records"
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            trim_audio(file_path)

    file_list = glob.glob("trans_records/*.wav")
    transcriptions = []
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2", device=device)

    for file_path in tqdm(file_list, desc="Transcribing files"):
        # Extract the module name from the file_name
        file_name = os.path.basename(file_path)
        module = ""
        if "Adult" in file_name:
            module = "Adult"
        elif "Child" in file_name:
            module = "Child"
        elif "Neonate" in file_name:
            module = "Neonate"

        p = pipe(file_path)
        print(p)
        text = p["text"]
        
        id_match = re.search(r"\d+", file_name)
        if id_match:
            id_name = id_match.group()
        else:
            id_name = "Unknown"

        transcriptions.append([id_name, module, text, "unknown"])

    column_names = ["ID", "module", "open_response", "LLM_CoD"]

    csv_file_path = "translated_script.csv"
    with open(csv_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(column_names)  # Write the column names as the first row
        writer.writerows(transcriptions)

    print(f"Transcriptions saved as {csv_file_path}.")


### Classify ###
def analyze_data(input_file, chosen_cause, output_file):
    available_models = [
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "facebook/bart-large-mnli",
        "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "cross-encoder/nli-distilroberta-base"
    ]
    print("Available models:")
    for i, model_name in enumerate(available_models):
        print(f"{i + 1}. {model_name}")

    model_choice = input("Enter the number corresponding to the desired model: ")
    model_index = int(model_choice) - 1

    if model_index < 0 or model_index >= len(available_models):
        print("Invalid model choice. Exiting.")
        return

    model = available_models[model_index]
    task = "zero-shot-classification"
    #model = "facebook/bart-large-mnli"
    device = 0 if torch.cuda.is_available() else -1

    tokenizer = AutoTokenizer.from_pretrained(model)
    zs_class = pipeline(task=task, model=model, device=device)

    first_round = ["Health Information", "Short no comment", "Thank you", "Cause of Death"]
    all_causes = ["Cirrhosis", "Epilepsy", "Pneumonia", "COPD", "Acute Myocardial Infarction", "Fires", "Renal Failure", "AIDS",
                  "Lung Cancer", "Maternal", "Drowning", "Other Cardiovascular Diseases", "Other Non-communicable Diseases", "Falls", "Road Traffic",
                  "Bite of Venomous Animal", "Diabetes", "Other Infectious Diseases", "TB", "Suicide", "Other Injuries", "Cervical Cancer", "Stroke",
                  "Malaria", "Asthma", "Colorectal Cancer", "Homicide", "Diarrhea/Dysentery", "Breast Cancer", "Leukemia/Lymphomas", "Poisonings", "Prostate Cancer",
                  "Esophageal Cancer", "Stomach Cancer", "Measles", "Other Defined Causes of Child Deaths", "Violent Death", "Other Digestive Diseases",
                  "Encephalitis", "Sepsis", "Other Cancers", "Hemorrhagic fever", "Meningitis", "Birth asphyxia", "Stillbirth", "Preterm Delivery", "Meningitis/Sepsis",
                  "Congenital malformation"]
    
    ICD_causes = ["K74.6", "G40", "J18", "J44", "I21.9", "X00-X09", "N19", "B20-B24",
              "C34", "O00-O99", "T75", "I00-I99", "A00-B99", "W00-W19", "V01-V99",
              "T63", "E10-E14", "A00-B99", "A16", "X60-X84", "V01-Y98", "C53",
              "I60-I69", "B50-B54", "J45", "C18-C21", "X85-Y09", "C50", "C81-C96",
              "X40-X49", "C61", "C15", "C16", "B05", "P00-P96", "Y35-Y36", "K92",
              "A80-A89", "A40-A41", "B00-B02", "C00-C96", "A97-A98", "G00-G09",
              "P20-P24", "P07", "P07.2", "G00-G09", "Q00-Q99"]

    neo_cause = ["Birth asphyxia", "Stillbirth", "Preterm Delivery", "Meningitis/Sepsis", "Congenital malformation", "Pneumonia"]
    child_cause = ["Bite of Venomous Animal", "Malaria", "Measles", "Pneumonia", "AIDS", "Other Defined Causes of Child Deaths",
                   "Violent Death", "Diarrhea/Dysentery", "Road Traffic", "Encephalitis", "Other Infectious Diseases", "Sepsis", 
                   "Drowning", "Other Cancers", "Hemorrhagic fever", "Fires", "Meningitis", "Other Cardiovascular Diseases",
                   "Other Digestive Diseases", "Falls", "Poisonings"]

    adult_cause = ["Cirrhosis", "Epilepsy", "Pneumonia", "COPD", "Acute Myocardial Infarction", "Fires", "Renal Failure", "AIDS", 
                   "Lung Cancer", "Maternal", "Drowning", "Other Cardiovascular Diseases", "Other Non-communicable Diseases", "Falls",
                   "Road Traffic", "Diabetes", "Other Infectious Diseases", "TB", "Suicide", "Other Injuries", "Cervical Cancer", "Stroke",
                   "Malaria", "Asthma", "Colorectal Cancer", "Homicide", "Diarrhea/Dysentery", "Breast Cancer", "Leukemia/Lymphomas",
                   "Poisonings", "Prostate Cancer", "Esophageal Cancer", "Stomach Cancer", "Bite of Venomous Animal"]
    
    adult_ICD_cause = ["K74.6", "G40", "J18", "J44", "I21.9", "X00-X09", "N19", "B20-B24",
               "C34", "O00-O99", "T75", "I00-I99", "A00-B99", "W00-W19",
               "V01-V99", "E10-E14", "A00-B99", "A16", "X60-X84", "V01-Y98",
               "C53", "I60-I69", "B50-B54", "J45", "C18-C21", "X85-Y09",
               "C50", "C81-C96", "X40-X49", "C61", "C15", "C16", "T63"]


    if chosen_cause == "First Round":
        causes = first_round
    elif chosen_cause == "All Causes":
        causes = all_causes
    elif chosen_cause == "Neonatal Causes":
        causes = neo_cause
    elif chosen_cause == "Child Causes":
        causes = child_cause
    elif chosen_cause == "Adult Causes":
        causes = adult_cause
    elif chosen_cause == "Adult ICD Causes":
        causes = adult_ICD_cause
    elif chosen_cause == "ICD Causes":
        causes = ICD_causes    
    else:
        print("Invalid chosen_cause option.")
        return

    ids = []
    group = []
    open_responses = []
    true_cause = []
    data_rows = []
    
    
    with open(input_file, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            open_response = row.get("open_response") or row.get("Response")
            open_responses.append(open_response)
            
            true_cause_value = row.get("gs_text34") or row.get("True_CoD") or "Unknown"
            true_cause.append(true_cause_value)


            temp_id = row.get("ID") or row.get("\ufeffnewid")
            ids.append(temp_id)
            
            tmp_group = row.get("module") or row.get("Group")
            group.append(tmp_group)

    ran_samp2 = ListDataset(open_responses)

    for out in tqdm(zs_class(ran_samp2, candidate_labels=causes)):
        sequence = out['sequence']
        labels = out['labels'][:3]
        scores = out['scores'][:3]

        row = {
            "ID": ids.pop(0),
            "Group": group.pop(0),
            "Response": sequence,
            "Label_1": labels[0],
            "Score_1": scores[0],
            "Label_2": labels[1],
            "Score_2": scores[1],
            "Label_3": labels[2],
            "Score_3": scores[2],
            "True_CoD": true_cause.pop(0)
        }

        data_rows.append(row)

    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["ID", "Group", "Response", "Label_1", "Score_1", "Label_2", "Score_2", "Label_3", "Score_3", "True_CoD"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data_rows)

    print("Data saved to", output_file)

### CSMF ####
all_causes = ["Cirrhosis", "Epilepsy", "Pneumonia", "COPD", "Acute Myocardial Infarction", "Fires", "Renal Failure", "AIDS",
                  "Lung Cancer", "Maternal", "Drowning", "Other Cardiovascular Diseases", "Other Non-communicable Diseases", "Falls", "Road Traffic",
                  "Bite of Venomous Animal", "Diabetes", "Other Infectious Diseases", "TB", "Suicide", "Other Injuries", "Cervical Cancer", "Stroke",
                  "Malaria", "Asthma", "Colorectal Cancer", "Homicide", "Diarrhea/Dysentery", "Breast Cancer", "Leukemia/Lymphomas", "Poisonings", "Prostate Cancer",
                  "Esophageal Cancer", "Stomach Cancer", "Measles", "Other Defined Causes of Child Deaths", "Violent Death", "Other Digestive Diseases",
                  "Encephalitis", "Sepsis", "Other Cancers", "Hemorrhagic fever", "Meningitis", "Birth asphyxia", "Stillbirth", "Preterm Delivery", "Meningitis/Sepsis",
                  "Congenital malformation"]

neo_cause = ["Birth asphyxia", "Stillbirth", "Preterm Delivery", "Meningitis/Sepsis", "Congenital malformation", "Pneumonia"]
child_cause = ["Bite of Venomous Animal", "Malaria", "Measles", "Pneumonia", "AIDS", "Other Defined Causes of Child Deaths",
                   "Violent Death", "Diarrhea/Dysentery", "Road Traffic", "Encephalitis", "Other Infectious Diseases", "Sepsis", 
                   "Drowning", "Other Cancers", "Hemorrhagic fever", "Fires", "Meningitis", "Other Cardiovascular Diseases",
                   "Other Digestive Diseases", "Falls", "Poisonings"]

adult_cause = ["Cirrhosis", "Epilepsy", "Pneumonia", "COPD", "Acute Myocardial Infarction", "Fires", "Renal Failure", "AIDS", 
                   "Lung Cancer", "Maternal", "Drowning", "Other Cardiovascular Diseases", "Other Non-communicable Diseases", "Falls",
                   "Road Traffic", "Diabetes", "Other Infectious Diseases", "TB", "Suicide", "Other Injuries", "Cervical Cancer", "Stroke",
                   "Malaria", "Asthma", "Colorectal Cancer", "Homicide", "Diarrhea/Dysentery", "Breast Cancer", "Leukemia/Lymphomas",
                   "Poisonings", "Prostate Cancer", "Esophageal Cancer", "Stomach Cancer", "Bite of Venomous Animal"]


def CSMF():
    # Ask for the CSV file name
    csv_files = [file for file in os.listdir() if file.endswith(".csv")]

    if not csv_files:
        print("No .csv files found in the current directory.")
        return

    # Display the available .csv files for selection
    print("Available .csv files:")
    for i, file in enumerate(csv_files, start=1):
        print(f"{i}. {file}")

    while True:
        try:
            file_choice = int(input("Select a .csv file by entering its number: "))
            if file_choice < 1 or file_choice > len(csv_files):
                raise ValueError("Invalid file choice. Please enter a valid number.")
            break
        except ValueError as e:
            print(e)

    file_name = csv_files[file_choice - 1]

    try:
        df = pd.read_csv(file_name)

        # Check if "gs_text34" or "True_CoD" columns exist
        if "gs_text34" in df.columns and "True_CoD" in df.columns:
            column_choice = input("Both 'gs_text34' and 'True_CoD' columns exist. Enter the column to use (gs_text34/True_CoD): ")
            if column_choice.lower() == "gs_text34":
                column_to_use = "gs_text34"
            elif column_choice.lower() == "true_cod":
                column_to_use = "True_CoD"
            else:
                print("Invalid column choice.")
                return
        elif "gs_text34" in df.columns:
            column_to_use = "gs_text34"
        elif "True_CoD" in df.columns:
            column_to_use = "True_CoD"
        else:
            print("Neither 'gs_text34' nor 'True_CoD' columns exist in the DataFrame.")
            return

        # Define the cause categories
        cause_categories = {
            "all": all_causes,
            "neonatal": neo_cause,
            "child": child_cause,
            "adult": adult_cause
        }

        # Ask for the cause category
        cause_category = input("Enter the cause category (All/Neonatal/Child/Adult): ").lower()

        # Check if the cause category is valid
        if cause_category not in cause_categories:
            print("Invalid cause category.")
            return

        # Filter the DataFrame based on the cause category and selected column
        causes = cause_categories[cause_category]
        filtered_df = df[df[column_to_use].isin(causes)]

        
        filtered_df = df[df[column_to_use].isin(causes) | df['Label_1'].isin(causes)]

        # Calculate the percentage composition of each value in True_CoD and LLM_CoD columns
        true_cod_composition = filtered_df[column_to_use].value_counts(normalize=True) * 100
        llm_cod_composition = filtered_df['Label_1'].value_counts(normalize=True) * 100

        # Create a new DataFrame with causes and their corresponding composition values
        composition_df = pd.DataFrame(index=causes, columns=[column_to_use, 'LLM_CoD'])
        composition_df[column_to_use] = true_cod_composition
        composition_df['LLM_CoD'] = llm_cod_composition

        # Sort the DataFrame based on the True_CoD composition in descending order
        composition_df.sort_values(by=column_to_use, ascending=False, inplace=True)

        # Plot the percentage composition
        fig, ax = plt.subplots(figsize=(10, 10))
        composition_df.plot(kind='barh', ax=ax)
        plt.title('Cause-Specific Mortality Fraction')
        plt.xlabel('Percentage Composition')
        plt.ylabel('Causes ({})'.format(cause_category.capitalize()))
        plt.legend([column_to_use, 'LLM_CoD'], loc='upper right')
        plt.tight_layout()    
    

        # Save the plot as a PDF file
        plt.savefig('CSMF_plot.pdf')

        # Display the plot
        plt.show()

    except FileNotFoundError:
        print("File not found. Please make sure the file exists and try again.")

# Main code
if __name__ == "__main__":
    while True:
        task = input("What task would you like to perform? (Record, Transcribe, Translate, Classify, CSMF, or Exit): ")

        if task.lower() == "record":
            recorded_file = record_audio()
        elif task.lower() == "transcribe":
            transcribe_files()
        elif task.lower() == "classify":
            #input_file = input("Enter the input file name: ")
            csv_files = [file for file in os.listdir() if file.endswith(".csv")]
            if not csv_files:
                print("No .csv files found in the current directory.")
                
            print("Available .csv files:")
            for i, file in enumerate(csv_files, start=1):
                print(f"{i}. {file}")

            while True:
                try:
                    file_choice = int(input("Select a .csv file by entering its number: "))
                    if file_choice < 1 or file_choice > len(csv_files):
                        raise ValueError("Invalid file choice. Please enter a valid number.")
                    break
                except ValueError as e:
                    print(e)

            input_file = csv_files[file_choice - 1]
            chosen_cause = input("Enter the chosen cause (All Causes, Neonatal Causes, Child Causes, Adult Causes): ")
            output_file = input("Enter the output file name: \n")

            if not output_file.endswith(".csv"):
                output_file += ".csv"

            analyze_data(input_file, chosen_cause, output_file)
        elif task.lower() == "csmf":
            CSMF()
        elif task.lower() == "translate":
            translate_files()
        elif task.lower() == "exit":
            break
        else:
            print("Invalid task. Please enter a valid task.")
