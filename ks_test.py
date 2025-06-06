import pandas as pd
import scipy.stats as stats
import argparse

def detect_drift_ks(original_file, drifted_file, text_col="Utterance"):
    print("🔍 Læser data...")
    df_orig = pd.read_csv("test_sent_emo.csv")
    df_drift = pd.read_csv("test_drifted.csv")

    # Ekstraher fx tekstlængde som numerisk feature
    print("📏 Ekstraherer tekstlængder...")
    orig_lengths = df_orig[text_col].astype(str).str.len()
    drift_lengths = df_drift[text_col].astype(str).str.len()

    print("📊 Udfører Kolmogorov–Smirnov-test...")
    ks_stat, p_value = stats.ks_2samp(orig_lengths, drift_lengths)

    print(f"\n📈 KS-statistik: {ks_stat:.4f}")
    print(f"📉 p-værdi: {p_value:.4f}")

    if p_value < 0.09:
        print("\n🚨 DRIFT DETEKTERET: Fordelingerne er signifikant forskellige.")
    else:
        print("\n✅ Ingen signifikant drift fundet i tekstlængde.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_file", type=str, required=True, help="Path til original test.csv")
    parser.add_argument("--drifted_file", type=str, required=True, help="Path til drifted test.csv")
    parser.add_argument("--text_col", type=str, default="text", help="Navnet på kolonnen med tekst")
    args = parser.parse_args()

    detect_drift_ks(args.original_file, args.drifted_file, args.text_col)
