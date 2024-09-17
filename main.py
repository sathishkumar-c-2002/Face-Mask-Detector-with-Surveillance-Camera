# main.py
import data_prep
import train_model
import detecting_mask

def main():
    print("[INFO] Data preparation...")
    data_prep.load_and_preprocess_data()
    
    print("[INFO] Training model...")
    train_model.train_and_save_model()
    
    print("[INFO] Detecting masks...")
    detecting_mask.detect_and_predict_mask()

if __name__ == "__main__":
    main()
