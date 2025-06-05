import sys
import io
import re
from carbontracker.tracker import CarbonTracker
from test import main as run_training_and_testing
from inference import run_single_inference

def extract_co2_from_output(output: str) -> float:
    match = re.search(r'CO2eq:\s+([\d\.]+)\s+g', output)
    if match:
        return float(match.group(1)) / 1000  # konverter til kg
    return 0.0

def main():
    # ---------- Træning og test ----------
    buffer = io.StringIO()
    sys.stdout = buffer  # midlertidig redirect af print

    tracker_train = CarbonTracker(epochs=1, log_dir='carbon_logs', monitor_epochs=1, update_interval=1)
    print("\n⚡ CarbonTracker starter måling for træning + test ...")
    tracker_train.epoch_start()

    run_training_and_testing()

    tracker_train.epoch_end()
    tracker_train.stop()

    sys.stdout = sys.__stdout__  # gendan print
    training_output = buffer.getvalue()
    print(training_output)  # vis hvad CarbonTracker loggede

    # ➕ Hent CO2 fra log
    training_emission = extract_co2_from_output(training_output)
    print(f"\n📊 Total CO2 for træning + test: {training_emission:.4f} kg CO2e")

    # ---------- Måling af én inferens ----------
    tracker_infer = CarbonTracker(epochs=1, log_dir='carbon_logs', monitor_epochs=1, update_interval=1)
    print("\n⚡ CarbonTracker måler én enkelt inferens ...")
    tracker_infer.epoch_start()

    run_single_inference()

    tracker_infer.epoch_end()
    tracker_infer.stop()

    # ➕ Fang output for inferens
    buffer = io.StringIO()
    sys.stdout = buffer

    run_single_inference()

    sys.stdout = sys.__stdout__
    infer_output = buffer.getvalue()
    print(infer_output)

    inference_emission = extract_co2_from_output(infer_output)
    print(f"📍 CO2-udledning for én forespørgsel: {inference_emission * 1000:.2f} g CO2e")

    # ---------- Estimat for årlig drift ----------
    requests_per_day = 100
    yearly_estimate = inference_emission * requests_per_day * 365
    print(f"\n📈 Estimeret årlig CO2-udledning ({requests_per_day} requests/dag):")
    print(f"    ≈ {yearly_estimate:.4f} kg CO2e")

if __name__ == "__main__":
    main()

