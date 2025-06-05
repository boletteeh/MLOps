# carbon_tracker_wrapper.py

from carbontracker.tracker import CarbonTracker
from test import main as run_training_and_testing
from inference import run_single_inference

def main():
    # ---------- Træning og test ----------
    tracker_train = CarbonTracker(epochs=1, log_dir='carbon_logs', monitor_epochs=1, update_interval=1)

    print("\n⚡ CarbonTracker starter måling for træning + test ...")
    tracker_train.epoch_start()

    run_training_and_testing()

    tracker_train.epoch_end()
    tracker_train.stop()

    print(f"\n📊 Total CO2 for træning + test: {training_emission:.4f} kg CO2e")

    # ---------- Måling af én inferens ----------
    tracker_infer = CarbonTracker(epochs=1, log_dir='carbon_logs', monitor_epochs=1, update_interval=1)
    print("\n⚡ CarbonTracker måler én enkelt inferens ...")
    tracker_infer.epoch_start()

    run_single_inference()

    tracker_infer.epoch_end()
    tracker_infer.stop()

    inference_emission = tracker_infer._emissions_tracker._total_emissions
    print(f"📍 CO2-udledning for én forespørgsel: {inference_emission * 1000:.2f} g CO2e")

    # ---------- Estimat for årlig drift ----------
    requests_per_day = 100  # Justér som ønsket
    yearly_estimate = inference_emission * requests_per_day * 365
    print(f"\n📈 Estimeret årlig CO2-udledning ({requests_per_day} requests/dag):")
    print(f"    ≈ {yearly_estimate:.4f} kg CO2e")

if __name__ == "__main__":
    main()
