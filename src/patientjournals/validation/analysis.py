import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


LABEL_ORDER = ["accept", "somewhat_accept", "reject", "unsure", "corrected"]
LABEL_DISPLAY = {
    "accept": "Accept",
    "somewhat_accept": "Somewhat Accept",
    "reject": "Reject",
    "unsure": "Unsure",
    "corrected": "Corrected",
}
LABEL_SCORES = {
    "accept": 1.0,
    "somewhat_accept": 1.0,
    "reject": 0.0,
    "unsure": 0.0,
    "corrected": 0.0,
}


def load_validations(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.rglob("*_validations.csv"))
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f"No validation files found at {path}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["label"].astype(str).str.lower()
    if "column_name" not in df.columns:
        raise ValueError("Missing column_name in validation data.")
    return df


def save_plot(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_label_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    counts = (
        df["label"]
        .value_counts()
        .reindex(LABEL_ORDER, fill_value=0)
        .rename_axis("label")
        .reset_index(name="count")
    )
    counts["label"] = counts["label"].map(LABEL_DISPLAY).fillna(counts["label"])

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=counts, x="label", y="count", hue="label", ax=ax, palette="viridis", legend=False)
    ax.set_title("Label Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    save_plot(fig, out_dir, "label_distribution")
    plt.close(fig)

def _add_accuracy_scores(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    scored["score"] = scored["label"].map(LABEL_SCORES)
    return scored.dropna(subset=["score"])


def plot_overall_accuracy(df: pd.DataFrame, out_dir: Path) -> None:
    scored = _add_accuracy_scores(df)
    if scored.empty:
        print("No scored rows found for overall accuracy plot.")
        return
    overall = scored["score"].mean()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(x=["Overall"], y=[overall], ax=ax, color=sns.color_palette("crest", 1)[0])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.set_title("Overall Accuracy")
    save_plot(fig, out_dir, "overall_accuracy")
    plt.close(fig)


def plot_top_level_accuracy(df: pd.DataFrame, out_dir: Path, min_n: int = 5) -> None:
    scored = _add_accuracy_scores(df)
    if scored.empty:
        print("No scored rows found for top-level accuracy plot.")
        return
    scored = scored.assign(top_level=scored["column_name"].astype(str).str.split(".").str[0])
    summary = (
        scored.groupby("top_level", as_index=False)
        .agg(n=("score", "size"), accuracy=("score", "mean"))
        .query("n >= @min_n")
        .sort_values("accuracy", ascending=False)
    )
    if summary.empty:
        print("No top-level columns met min_n for accuracy plots.")
        return
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=summary,
        x="top_level",
        y="accuracy",
        hue="top_level",
        ax=ax,
        palette="crest",
        legend=False,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.set_title("Top-Level Accuracy")
    ax.tick_params(axis="x", rotation=20)
    save_plot(fig, out_dir, "top_level_accuracy")
    plt.close(fig)


def plot_nested_accuracy(df: pd.DataFrame, out_dir: Path, min_n: int = 5) -> None:
    scored = _add_accuracy_scores(df)
    if scored.empty:
        print("No scored rows found for nested accuracy plots.")
        return
    scored = scored.assign(top_level=scored["column_name"].astype(str).str.split(".").str[0])
    for top_level, subset in scored.groupby("top_level"):
        summary = (
            subset.groupby("column_name", as_index=False)
            .agg(n=("score", "size"), accuracy=("score", "mean"))
            .query("n >= @min_n")
            .sort_values("accuracy", ascending=False)
        )
        if summary.empty:
            continue
        fig_height = max(4, min(18, 0.35 * len(summary)))
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(9, fig_height))
        sns.barplot(
            data=summary,
            y="column_name",
            x="accuracy",
            hue="column_name",
            ax=ax,
            palette="crest",
            legend=False,
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("")
        ax.set_title(f"Accuracy for {top_level} Columns")
        safe_name = top_level.replace("/", "_")
        save_plot(fig, out_dir, f"nested_accuracy_{safe_name}")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple validation analysis with seaborn.")
    parser.add_argument("--input", required=True, help="Validation CSV file or folder")
    parser.add_argument("--out", default="validation_reports", help="Output folder for plots")
    parser.add_argument("--min-n", type=int, default=1, help="Min samples per column for inclusion")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)

    df = load_validations(input_path)
    plot_label_distribution(df, out_dir)
    plot_overall_accuracy(df, out_dir)
    plot_top_level_accuracy(df, out_dir, min_n=args.min_n)
    plot_nested_accuracy(df, out_dir, min_n=args.min_n)

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
