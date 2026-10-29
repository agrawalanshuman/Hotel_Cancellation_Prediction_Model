from pathlib import Path
import os, re
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('DATA_DIR', ROOT / 'data/raw'))
TEAM_NAME = os.environ.get('TEAM_NAME', 'Ctrl_+_Alt_+_Defeat')
SEED = 835  # exact professor split; use consistently across comparisons
TARGET = 'is_canceled'
TRAIN_ROWS, TEST_ROWS = 83573, 35817
KNOWN_LEAKS = ('reservation_status', 'reservation_status_date', 'assigned_room_type')
UNCERTAIN = ('booking_changes', 'days_in_waiting_list', 'total_of_special_requests', 'agent', 'company', 'country', 'deposit_type')
def team_name():
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_+\-]{0,79}', TEAM_NAME):
        raise ValueError('Set TEAM_NAME to the verified official name (letters, digits, underscore, plus, hyphen).')
    return TEAM_NAME
