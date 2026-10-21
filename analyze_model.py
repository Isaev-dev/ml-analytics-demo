#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Разбор готовой master_db: важности признаков, acc±1, скрининговый 3-классовый режим.
Пути строятся от расположения скрипта — запускать можно из любой папки."""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MFILE = os.path.join(ROOT, "out", "master_db.csv")
REPORT = os.path.join(ROOT, "out", "analyze_report.txt")

FEAT = ["MW", "LogP", "TPSA", "nDon", "nAcc", "nRot", "nRings", "nArom", "Fsp3",
        "MR", "nHeavy", "BalabanJ", "BertzCT", "Chi1v", "Kappa2", "EState",
        "reeig11", "belm2", "belm2_max", "H_atoms", "H_bonds"]


def main():
    if not os.path.exists(MFILE):
        raise SystemExit(f"Нет {MFILE}. Сначала: python src/build_db.py --toxval --reuse")
    df = pd.read_csv(MFILE)
    missing = [c for c in FEAT + ["gost4"] if c not in df.columns]
    if missing:
        raise SystemExit(f"В master_db нет колонок: {missing}")

    X = df[FEAT].values
    y = df["gost4"].astype(int).values
    lines = [f"объектов: {len(df)}; классы: {dict(zip(*np.unique(y, return_counts=True)))}"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y,
                                          random_state=42)
    rf = make_pipeline(SimpleImputer(strategy="median"),
                       RandomForestClassifier(500, class_weight="balanced",
                                              random_state=42, n_jobs=-1))
    rf.fit(Xtr, ytr)
    p = rf.predict(Xte)
    lines.append(f"4 класса RF-bal: acc={accuracy_score(yte, p):.3f} "
                 f"bal_acc={balanced_accuracy_score(yte, p):.3f} "
                 f"f1_macro={f1_score(yte, p, average='macro'):.3f} "
                 f"acc±1 класс={np.mean(np.abs(p - yte) <= 1):.3f}")
    lines.append("матрица ошибок (строки=истина, столбцы=прогноз):")
    lines.append(str(confusion_matrix(yte, p, labels=[1, 2, 3, 4])))

    imp = pd.Series(rf.named_steps["randomforestclassifier"].feature_importances_,
                    index=FEAT).sort_values(ascending=False)
    lines.append("Топ-15 важностей признаков:")
    lines.append(imp.head(15).to_string())

    # Скрининговый режим: блок 1+2 (высокоопасные) / 3 (умеренные) / 4 (малоопасные)
    y3tr = np.where(ytr <= 2, 1, ytr - 1)
    y3te = np.where(yte <= 2, 1, yte - 1)
    rf3 = make_pipeline(SimpleImputer(strategy="median"),
                        RandomForestClassifier(500, class_weight="balanced",
                                               random_state=42, n_jobs=-1))
    rf3.fit(Xtr, y3tr)
    p3 = rf3.predict(Xte)
    lines.append(f"3 класса (блок 1+2 / умеренные / малоопасные): "
                 f"acc={accuracy_score(y3te, p3):.3f} "
                 f"bal_acc={balanced_accuracy_score(y3te, p3):.3f} "
                 f"f1_macro={f1_score(y3te, p3, average='macro'):.3f}")
    high = y3te == 1
    sens = float(np.mean(p3[high] == 1)) if high.any() else float("nan")
    lines.append(f"скрининговая чувствительность к высокоопасному блоку (1+2): {sens:.3f}")

    txt = "\n".join(lines)
    print(txt)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(f"\nсохранено: {REPORT}")


if __name__ == "__main__":
    main()