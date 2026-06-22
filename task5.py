import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import folium
except ImportError:
    folium = None


def load_data(path):
    return pd.read_csv(path)


def preprocess(df, datetime_column="Date"):
    if datetime_column in df.columns:
        df[datetime_column] = pd.to_datetime(df[datetime_column], errors="coerce")
        df["hour"] = df[datetime_column].dt.hour
        df["day_of_week"] = df[datetime_column].dt.day_name()
        bins = [-1, 5, 11, 17, 23]
        labels = ["Night", "Morning", "Afternoon", "Evening"]
        df["time_of_day"] = pd.cut(df["hour"].fillna(-1), bins=bins, labels=labels)
        df["time_of_day"] = df["time_of_day"].cat.add_categories(["Unknown"]).fillna("Unknown")
    else:
        df["hour"] = np.nan
        df["day_of_week"] = "Unknown"
        df["time_of_day"] = "Unknown"

    for col in ["Road_Condition", "Weather_Condition", "time_of_day"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.title().replace("Nan", "Unknown")

    return df


def summarize_factors(df):
    factors = {}
    for col in ["Road_Condition", "Weather_Condition", "time_of_day"]:
        if col in df.columns:
            summary = df[col].fillna("Unknown").value_counts().rename_axis(col).reset_index(name="count")
            factors[col] = summary
    return factors


def plot_counts(factors, output_dir):
    sns.set(style="whitegrid")
    for col, summary in factors.items():
        plt.figure(figsize=(10, 6))
        sns.barplot(data=summary, x="count", y=col, hue=col, palette="viridis", legend=False)
        plt.title(f"Accident count by {col.replace('_', ' ').title()}")
        plt.xlabel("Accident Count")
        plt.ylabel(col.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_counts.png"), dpi=150)
        plt.close()


def plot_time_heatmap(df, output_dir):
    if "hour" not in df.columns or "day_of_week" not in df.columns:
        return
    heat = df.pivot_table(index="day_of_week", columns="hour", values="Road_Condition", aggfunc="count")
    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat = heat.reindex(ordered_days).fillna(0)
    plt.figure(figsize=(12, 6))
    sns.heatmap(heat, annot=True, fmt=".0f", cmap="mako", cbar_kws={"label": "Accident Count"})
    plt.title("Accident frequency by day of week and hour")
    plt.ylabel("Day of Week")
    plt.xlabel("Hour of Day")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_of_day_heatmap.png"), dpi=150)
    plt.close()


def plot_hotspots(df, output_dir):
    if folium is None:
        return
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        return
    valid = df.dropna(subset=["Latitude", "Longitude"])
    if valid.empty:
        return
    center = [valid["Latitude"].mean(), valid["Longitude"].mean()]
    accident_map = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
    for _, row in valid.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=2,
            color="red",
            fill=True,
            fill_opacity=0.4,
        ).add_to(accident_map)
    accident_map.save(os.path.join(output_dir, "accident_hotspots.html"))


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "traffic_accidents.csv"
    output_dir = "analysis_outputs"
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(csv_path):
        print(f"Input file not found: {csv_path}")
        return
    df = load_data(csv_path)
    df = preprocess(df)
    factors = summarize_factors(df)
    plot_counts(factors, output_dir)
    plot_time_heatmap(df, output_dir)
    plot_hotspots(df, output_dir)
    print(f"Analysis complete. Charts saved to {output_dir}")
    if folium is None:
        print("Install folium to generate hotspot map output.")


if __name__ == "__main__":
    main()
