# Expected-Goals-Modell (WM 2022)

Code zur Bachelorarbeit **„Entwicklung eines Expected-Goals-Modells zur Quantifizierung von Torwahrscheinlichkeiten anhand von X/Y-Event-Daten“** (B.Sc. Informatik, IU Internationale Hochschule, 2026).

Die Pipeline lädt die Schussdaten der FIFA-Weltmeisterschaft 2022 aus den StatsBomb Open Data, berechnet geometrische Merkmale (Distanz, Sichtwinkel), trainiert drei logistische Regressionsmodelle und vergleicht sie mit zwei Baselines sowie dem StatsBomb-xG-Modell.

## Inhalt der Pipeline

| Schritt | Beschreibung |
|---|---|
| Datenabruf | StatsBomb Open Data über `statsbombpy` (Wettbewerb 43, Saison 106, 64 Spiele), lokales Caching als `shots_wm2022.csv` |
| Bereinigung | Filter auf Schüsse, Ausschluss von Elfmetern (1430 Schüsse) |
| Feature Engineering | Distanz zum Torzentrum (Pythagoras, Yards → Meter), sichtbarer Schusswinkel (Kosinussatz, Grad), One-Hot-Encoding von Körperteil und Spielsituation |
| Split | Nach Spielen gruppierter 80/20-Split (`GroupShuffleSplit`, Seed 8): 51 Trainings-, 13 Testspiele |
| Modelle | Basisquote, Distanz-Heuristik (16.5 m), Distanz-Regression, LR (Geometrie), LR (Erweitert) |
| Tuning | `LogisticRegressionCV` mit `StratifiedGroupKFold` (5 Folds), L2-Regularisierung |
| Inferenz | Unregularisierte Logit-Regression (statsmodels) mit nach Spiel geclusterten Standardfehlern |
| Evaluation | Log-Loss, ROC-AUC, Brier Score mit gruppiertem Bootstrap (1000 Resamples, `random_state=42`), Kalibrierungsprüfung, Vergleich mit `shot_statsbomb_xg` |

## Installation

```bash
git clone https://github.com/thedl003/bachelorarbeit-xg.git
cd bachelorarbeit-xg
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
pip install -r requirements.txt
```

## Ausführung

```bash
python xg_model_base.py
```

Beim ersten Lauf werden die Event-Daten aller 64 Spiele heruntergeladen (mehrere Minuten). Danach wird der lokale Cache `shots_wm2022.csv` verwendet; die Metadaten des Abrufs (Zeitpunkt, statsbombpy-Version, Kennzahlen) werden daneben als JSON gespeichert.

Erzeugte Dateien:

| Datei | Inhalt |
|---|---|
| `shots_wm2022.csv` | Bereinigter Schuss-Datensatz (Cache, beim ersten Lauf erzeugt) |
| `shots_wm2022_meta.json` | Metadaten des Datenabrufs (beim ersten Lauf erzeugt) |
| `shotmap_export.csv` | Testschüsse mit Koordinaten, Distanz, Winkel, Torerfolg und xG des erweiterten Modells (Grundlage für Abb. 2, 4 und 5) |

## Reproduzierbarkeit

- **Datenstand:** StatsBomb Open Data, Commit [`4b73468`](https://github.com/statsbomb/open-data/tree/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb). `statsbombpy` lädt immer den aktuellen Stand des `master`-Branches. Ändert StatsBomb die WM-2022-Daten nachträglich, können die Ergebnisse abweichen.
- **Seeds:** Split `seed=8`, Bootstrap `random_state=42`.
- **Softwareversionen:** siehe `requirements.txt` (Python 3.13.5).

## Ergebnisse (Testdatensatz, n = 258 Schüsse aus 13 Spielen)

| Modell | Log-Loss ↓ | ROC-AUC ↑ | Brier Score ↓ |
|---|---|---|---|
| Basisquote | 0.3270 | 0.5000 | 0.0907 |
| Heuristik (16.5 m) | 0.2992 | 0.6905 | 0.0858 |
| Distanz-Regression | 0.2701 | 0.8075 | 0.0771 |
| LR (Geometrie) | 0.2610 | 0.8098 | 0.0732 |
| LR (Erweitert) | 0.2593 | 0.8006 | 0.0719 |
| StatsBomb (Benchmark) | 0.2494 | 0.8183 | 0.0693 |

Konfidenzintervalle und Interpretation siehe Kapitel 5 der Arbeit.

## Datenquelle

Die Daten stammen aus den [StatsBomb Open Data](https://github.com/statsbomb/open-data) und unterliegen deren Nutzungsbedingungen. Rohdaten sind nicht Teil dieses Repositories.

## Autor

Tobias Hedl
