#!/usr/bin/env python3
import os
import pandas as pd

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

def load_and_clean(data_dir="data"):
    train_path = os.path.join(data_dir, "adult.data")
    test_path = os.path.join(data_dir, "adult.test")

    train = pd.read_csv(
        train_path,
        names=COLUMNS,
        sep=",",
        na_values=["?"],
        skipinitialspace=True,
        header=None,
        engine="python",
    )

    test = pd.read_csv(
        test_path,
        names=COLUMNS,
        sep=",",
        na_values=["?"],
        skipinitialspace=True,
        header=None,
        skiprows=1,
        engine="python",
    )

    test["income"] = test["income"].astype(str).str.strip().str.replace(r"\.$", "", regex=True)
    train["income"] = train["income"].astype(str).str.strip()

    df = pd.concat([train, test], ignore_index=True)

    df = df.dropna().reset_index(drop=True)

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["education-num"] = pd.to_numeric(df["education-num"], errors="coerce")
    df["capital-gain"] = pd.to_numeric(df["capital-gain"], errors="coerce")
    df["capital-loss"] = pd.to_numeric(df["capital-loss"], errors="coerce")
    df["hours-per-week"] = pd.to_numeric(df["hours-per-week"], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    return df

def prepare(df):
    df["y"] = (df["income"] == ">50K").astype(int)
    df["sex_male"] = (df["sex"] == "Male").astype(int)
    return df

def outcome_distribution_by_sex(df):
    return df.groupby(["sex", "income"]).size().unstack(fill_value=0)

def mean_diff_positive_by_sex(df):
    rates = df.groupby("sex")["y"].mean()
    male = rates.get("Male", float("nan"))
    female = rates.get("Female", float("nan"))
    return male - female

def correlation_y_sex(df):
    return df["y"].corr(df["sex_male"])

def main():
    df = load_and_clean("data")
    df = prepare(df)

    counts = outcome_distribution_by_sex(df)
    print("Outcome distribution (counts) by sex")
    print(counts.to_string())

    md = mean_diff_positive_by_sex(df)
    print("Mean difference in positive outcome rate (Male - Female):", md)

    corr = correlation_y_sex(df)
    print("Correlation(y, sex_male):", corr)

if __name__ == "__main__":
    main()