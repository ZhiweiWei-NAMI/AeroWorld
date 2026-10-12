"""Fixed L3 simulation and UE export entry."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l3_ue_delivery import load_spec, deliver_config
CONFIG = 'interventions/L3-3_v2/seed01/grp_pedestrian_evacuation__crowd_evacuation_repaired/arm_factual.yaml'
SPEC = load_spec(ROOT / CONFIG)
if __name__ == "__main__":
    print(deliver_config(ROOT / CONFIG), flush=True)
