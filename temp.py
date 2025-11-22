import os

# Paths
data_dir = r'C:\Users\rlaal\Documents\NUS\AY2526S1\CS3244\Project\osic-pulmonary-fibrosis-progression'
train_dir = os.path.join(data_dir, 'train')
preprocessed_dir = os.path.join(data_dir, 'preprocessed_scans')

# Get all patient folders in train
train_patients = set([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
print(f"Total patients in train: {len(train_patients)}")

# Get all patient folders in preprocessed_scans
if os.path.exists(preprocessed_dir):
    preprocessed_patients = set([d for d in os.listdir(preprocessed_dir) if os.path.isdir(os.path.join(preprocessed_dir, d))])
    print(f"Total patients in preprocessed_scans: {len(preprocessed_patients)}")
    
    # Find missing patients
    missing = train_patients - preprocessed_patients
    print(f"\nPatients NOT preprocessed: {len(missing)}")
    if missing:
        print("Missing patients:")
        for p in sorted(list(missing))[:20]:
            print(f"  {p}")
else:
    print("preprocessed_scans directory does not exist!")
    missing = train_patients

# Check specific patient
problem_patient = "ID00011637202177653955184"
if problem_patient in missing:
    print(f"\n✓ Confirmed: {problem_patient} is NOT in preprocessed_scans")
    print(f"  Raw scans exist: {os.path.exists(os.path.join(train_dir, problem_patient))}")
    if os.path.exists(os.path.join(train_dir, problem_patient)):
        dcm_count = len([f for f in os.listdir(os.path.join(train_dir, problem_patient)) if f.endswith('.dcm')])
        print(f"  Number of .dcm files: {dcm_count}")
