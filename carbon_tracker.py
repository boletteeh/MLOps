from carbontracker.tracker import CarbonTracker
from test import main as run_training_and_testing

def main():
    # Initialisér CarbonTracker for hele kørsel (træning + test)
    tracker = CarbonTracker(epochs=1, log_dir='carbon_logs', monitor_epochs=1, update_interval=1)

    tracker.epoch_start()  # Start måling
    run_training_and_testing()  # Kald din combined træning/test funktion
    tracker.epoch_end()  # Stop måling

if __name__ == "__main__":
    main()
