# convert_pt_to_zip.py
import os
import glob
from stable_baselines3 import PPO  # Change to your actual agent class if needed

# Set this to your agent directory
agent_dir = "./saved_models/CSTR1/DemandResponse/"

pt_files = glob.glob(os.path.join(agent_dir, "agent_*.pt"))

for pt_file in pt_files:
    zip_file = pt_file[:-3] + ".zip"
    print(f"Converting {pt_file} -> {zip_file}")
    try:
        # Try loading as SB3 model
        model = PPO.load(pt_file)
        model.save(zip_file)
        print(f"Saved {zip_file}")
    except Exception as e:
        print(f"Failed to convert {pt_file}: {e}")