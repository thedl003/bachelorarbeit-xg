import math
import os
import json
from datetime import datetime
from importlib.metadata import version
import pandas as pd
import numpy as np
from statsbombpy import sb
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

# Konstanten für die Geometrie des Spielfelds (StatsBomb-Koordinatensystem)
GOAL_X = 120.0
GOAL_POST_Y1 = 36.0 
GOAL_POST_Y2 = 44.0
GOAL_CENTER_Y = 40.0

def calculate_distance(x, y):
    """
    Berechnen der Distanz (Luftlinie) vom Schusspunkt (x, y) 
    zur Mitte des Tores (GOAL_X, GOAL_CENTER_Y).
    StatsBomb-Koordinaten sind in Yards, das Ergebnis wird in Meter umgerechnet.
    """
    x_diff = GOAL_X - x
    y_diff = GOAL_CENTER_Y - y
    return math.sqrt(x_diff**2 + y_diff**2) * 0.9144 # Umrechnung Yards -> Meter (1 Yard = 0.9144 m)

def calculate_angle(x, y):
    """
    Berechnen des Sichtwinkels auf das Tor [Grad], also des Winkels am Schusspunkt
    zwischen den Linien zu den beiden Pfosten, mithilfe des Kosinussatzes im
    Dreieck (Schusspunkt - Pfosten 1 - Pfosten 2).
    """
    a = math.sqrt((GOAL_X - x)**2 + (GOAL_POST_Y1 - y)**2) # Berechnen der Luftlinie zu Pfosten 1
    b = math.sqrt((GOAL_X - x)**2 + (GOAL_POST_Y2 - y)**2) # Berechnen der Luftlinie zu Pfosten 2
    c = abs(GOAL_POST_Y2 - GOAL_POST_Y1) #Wir nutzen abs um den Betrag zu erhalten
    
    if a == 0 or b == 0: # Dann steht der Schütze exakt auf einem der beiden Pfosten (Winkel undefiniert)
        return 0

    cos_alpha = (a**2 + b**2 - c**2) / (2 * a * b) # Nutzen des Kosinussatzes
    cos_alpha = max(-1.0, min(1.0, cos_alpha)) # Sicherstellen, dass durch Rundung kein Fehler entsteht
    return math.degrees(math.acos(cos_alpha))

def fetch_season_shots(competition_id=43, season_id=106): # Das sind die internen IDs für die WM 2022 der Männer
    """
    Lädt alle Spiele einer Saison (Standard: WM 2022) herunter,
    filtert Schüsse und bereinigt die Daten.
    """
    print(f"Lade Spielplan für Wettbewerb {competition_id}, Saison {season_id}...")
    matches = sb.matches(competition_id=competition_id, season_id=season_id) # Aufruf von statsbombpy (ohne Zugangsdaten werden die Open Data aus dem GitHub-Repo statsbomb/open-data geladen), Ergebnis als Pandas DF
    match_ids = matches['match_id'].tolist() # Extrahieren der Match_IDs aus dem Pandas DF
    print(f"Gefundene Spiele: {len(match_ids)}. Starte Download...")

    all_shots = [] # Arbeiten mit einer Liste um Performance zu sichern (DF wäre in einer Schleife zu langsam)
    total_events_count = 0
    with_penalty = 0
    skipped_no_loc = 0 # Zähler: Schüsse ohne gültige Koordinaten
    skipped_games_exception = 0 # Zähler: Spiele, deren Verarbeitung mit einem Fehler abgebrochen ist
    for n, m_id in enumerate(match_ids, 1): # Nutzen von Enumerate um Zählervariable zu vermeiden und um bei 1 zu starten
        try:
            events = sb.events(match_id=m_id) # Laden aller Events in einen DF
            total_events_count += len(events)
            if 'type' not in events.columns or 'Shot' not in events['type'].unique(): # Überspringen des Matches wenn es keinen Schuss gab
                continue

            # Data Filter 1: Behalten von Schüssen
            shots_df = events[events['type'] == 'Shot'].copy() # Kopieren von Events die vom Typ "Shot" sind 
            with_penalty += len(shots_df)
            # Data Filter 2: Herausfiltern von Elfmetern
            if 'shot_type' in shots_df.columns: # Aus unserer neuen Tabelle den Typen des "Shot" extrahieren
                shots_df = shots_df[shots_df['shot_type'] != 'Penalty'] # Behalten von Schüssen die kein Elfmeter sind

            # Iteriere durch jeden Schuss und extrahiere Features
            for _, row in shots_df.iterrows(): # Durchlaufen jeder Zeile mithilfe von iterrows
                location = row.get('location')
                if not isinstance(location, list) or len(location) < 2: # Überspringen des Schusses, wenn es nur eine oder keine Koordinate gibt
                    skipped_no_loc += 1
                    continue

                x = location[0] # Speichern der Koordinaten
                y = location[1]
                dist = calculate_distance(x, y) # Berechnen von Distanz und Winkel
                angle = calculate_angle(x, y)

                # Data Filter 3: Binäres Label (1 für Tor, 0 für alles andere)
                outcome = str(row.get('shot_outcome', '')).strip() # Erstellen eines Strings "Outcome" mit "Goal" oder nicht
                if outcome == 'Goal': 
                    is_goal = 1  
                else: 
                    is_goal = 0

                #Speichern von verschiedenen Metadaten zu dem Schuss
                body_part = str(row.get('shot_body_part', 'Unknown')) 
                play_pattern = str(row.get('play_pattern', 'Unknown'))
                statsbomb_xg = row.get('shot_statsbomb_xg', np.nan)

                all_shots.append({ # Erstellen einer Liste von Dictionaries mit den Metadaten
                    'match_id': m_id,
                    'player': str(row.get('player', 'Unknown')),
                    'x': x,
                    'y': y,
                    'distance_to_goal': dist,
                    'visible_angle': angle,
                    'shot_body_part': body_part,
                    'play_pattern': play_pattern,
                    'is_goal': is_goal,
                    'statsbomb_xg': statsbomb_xg
                })

            if n % 10 == 0 or n == len(match_ids): # Informieren des Users über den Fortschritt im Terminal immer nach 10 Matches oder nach Fertigstellung (verhindern von Spam)
                print(f"Progress: {n}/{len(match_ids)} Spiele verarbeitet...")

        except Exception as e:
            skipped_games_exception += 1
            print(f"Fehler bei Spiel {m_id}: {e}")
    df = pd.DataFrame(all_shots) # Erstellen eines DF um sauber weiterarbeiten zu können (Liste nur für Performance Zwecke)

    # Kennzahlen der Extraktion, werden neben der CSV gespeichert, damit sie auch bei Nutzung des Caches ausgegeben werden können
    stats = {
        'abrufzeitpunkt': datetime.now().isoformat(timespec='seconds'),
        'statsbombpy_version': version('statsbombpy'),
        'competition_id': competition_id,
        'season_id': season_id,
        'spiele': len(match_ids),
        'roh_events': total_events_count,
        'schuesse_inkl_elfmeter': with_penalty,
        'schuesse_extrahiert': len(df),
        'uebersprungen_keine_koordinaten': skipped_no_loc,
        'uebersprungen_spiele_mit_fehler': skipped_games_exception,
    }
    return df, stats

def print_extraction_stats(stats):
    """
    Ausgabe der beim Download gespeicherten Kennzahlen der Datenextraktion (Reproduzierbarkeit).
    """
    print(f"Datenabruf: {stats['abrufzeitpunkt']} (statsbombpy {stats['statsbombpy_version']}, "
          f"Wettbewerb {stats['competition_id']}, Saison {stats['season_id']})")
    print(f"Gesamtzahl aller Roh-Events: {stats['roh_events']}")
    print(f"Gesamtanzahl an Schüssen (inkl. Elfmeter): {stats['schuesse_inkl_elfmeter']}")
    print(f"Erfolgreich extrahiert (ohne Elfmeter): {stats['schuesse_extrahiert']} Schüsse aus {stats['spiele']} Spielen!")
    print(f"Übersprungen (keine Koordinaten): {stats['uebersprungen_keine_koordinaten']}")
    print(f"Übersprungen (Spiele mit Fehler): {stats['uebersprungen_spiele_mit_fehler']}")

def verify_data_quality(df):
    """
    Überprüfen der Qualität und Balance der extrahierten Schussdaten.
    """
    print("\n--- DATA PROFILING & QUALITÄTSCHECK ---")
    print(f"Gesamtanzahl Schüsse: {len(df)}")
    print(f"Tore (1): {df['is_goal'].sum()} ({(df['is_goal'].mean()*100):.2f}%)")
    print(f"Fehlschüsse (0): {(df['is_goal'] == 0).sum()} ({((1-df['is_goal'].mean())*100):.2f}%)")
    print("Fehlende Werte (NaNs) in primären Features:")
    print(df[['distance_to_goal', 'visible_angle', 'is_goal']].isnull().sum())
    print("---------------------------------------\n")

def grouped_train_test_split(y, groups, test_size=0.2, target_rate=None, max_tries=200, tolerance=0.01):
    """
    Split auf SPIEL-Ebene (alle Schüsse eines Spiels landen komplett in
    Training ODER Test), statt auf Schuss-Ebene. Verhindert,
    dass Schüsse desselben Spiels gleichzeitig in Training und Test
    vorkommen (Datenleck). Da sklearns train_test_split stratify nicht
    zusammen mit groups unterstützt, wird stattdessen GroupShuffleSplit
    verwendet und mehrfach mit unterschiedlichen Seeds gesplittet, bis die
    Torquote in Training und Test nah am Gesamtdurchschnitt liegt.
    """
    if target_rate is None:
        target_rate = y.mean()

    best_seed, best_diff, best_split = None, float("inf"), None

    for seed in range(max_tries):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y, groups=groups))
        train_rate = y.iloc[train_idx].mean()
        test_rate = y.iloc[test_idx].mean()
        diff = abs(train_rate - target_rate) + abs(test_rate - target_rate)

        if diff < best_diff:
            best_diff, best_seed, best_split = diff, seed, (train_idx, test_idx)

        if abs(train_rate - target_rate) <= tolerance and abs(test_rate - target_rate) <= tolerance:
            break

    train_idx, test_idx = best_split
    print(f"Gruppierter Split gewählt (Seed={best_seed}): "
          f"Train-Torquote={y.iloc[train_idx].mean()*100:.2f}%, "
          f"Test-Torquote={y.iloc[test_idx].mean()*100:.2f}%, "
          f"Spiele Train={groups.iloc[train_idx].nunique()}, Spiele Test={groups.iloc[test_idx].nunique()}")
    return train_idx, test_idx


def report_cv_performance(model, model_name):
    """
    Gibt die Cross-Validation-Performance (Log-Loss pro Fold) für das von
    LogisticRegressionCV gewählte C explizit aus, damit die Kreuzvalidierung
    als Ergebnis in der Arbeit selbst dokumentiert ist und nicht nur als
    interner Zwischenschritt für das C-Tuning dient.
    """
    scores = model.scores_[:, 0, :] # Form (n_folds, n_l1_ratios, n_cs), nur ein l1_ratio (0 = L2)
    best_c_idx = np.where(model.Cs_ == model.C_)[0][0]
    fold_scores = -scores[:, best_c_idx]
    print(f"CV-Performance ({model_name}, gewähltes C={model.C_:.4f}, "
          f"{len(fold_scores)} Folds): Log-Loss = {fold_scores.mean():.4f} ± {fold_scores.std():.4f}")
    return fold_scores


def bootstrap_ci(y_true, y_pred, groups, n_boot=1000, alpha=0.05, random_state=42):
    """
    95%-Bootstrap-Konfidenzintervalle für Log-Loss, ROC-AUC und Brier Score
    per gruppiertem Bootstrap auf Spielebene (ganze Spiele mit Zurücklegen
    gezogen, nicht einzelne Schüsse) - respektiert dieselbe Gruppenstruktur
    wie der Train-Test-Split.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    log_losses, aucs, briers = [], [], []

    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled_groups])
        yt, yp = y_true[idx], y_pred[idx]

        if len(np.unique(yt)) < 2:
            continue

        log_losses.append(log_loss(yt, yp))
        aucs.append(roc_auc_score(yt, yp))
        briers.append(brier_score_loss(yt, yp))

    def _ci(values):
        lower = np.percentile(values, 100 * alpha / 2)
        upper = np.percentile(values, 100 * (1 - alpha / 2))
        return float(np.mean(values)), float(lower), float(upper)

    ll, auc, br = _ci(log_losses), _ci(aucs), _ci(briers)

    print(f"  (n={len(aucs)} gültige von {n_boot} Resamples)")
    print(f"  Log-Loss: {ll[0]:.4f}  95%-KI [{ll[1]:.4f}, {ll[2]:.4f}]")
    print(f"  ROC-AUC:  {auc[0]:.4f}  95%-KI [{auc[1]:.4f}, {auc[2]:.4f}]")
    print(f"  Brier:    {br[0]:.4f}  95%-KI [{br[1]:.4f}, {br[2]:.4f}]")

    return {'log_loss': ll, 'roc_auc': auc, 'brier': br}


def calibration_check(y_true, y_pred, model_name, n_bins=5):
    """
    Prüft die Kalibrierung eines Modells: Stimmen die vorhergesagten
    Wahrscheinlichkeiten im Mittel mit der tatsächlich beobachteten
    Torquote überein? Dazu werden die Schüsse nach ihrem vorhergesagten
    xG-Wert in n_bins gleich große Gruppen eingeteilt (pd.qcut) und pro
    Gruppe der Mittelwert der Vorhersage mit der tatsächlichen Torquote
    verglichen. Ergänzend: Summe aller vorhergesagten xG-Werte vs.
    tatsächliche Toranzahl über den gesamten Testdatensatz
    ("Kalibrierung im Großen").
    """
    df_cal = pd.DataFrame({'y_true': np.asarray(y_true), 'y_pred': np.asarray(y_pred)})
    df_cal['bin'] = pd.qcut(df_cal['y_pred'], q=n_bins, duplicates='drop')

    cal_table = df_cal.groupby('bin', observed=True).agg(
        n=('y_true', 'size'),
        mean_predicted=('y_pred', 'mean'),
        actual_rate=('y_true', 'mean')
    ).reset_index()

    print(f"\n--- KALIBRIERUNG: {model_name} ({n_bins} Bins) ---")
    print(cal_table.to_string(index=False))

    total_pred_goals = df_cal['y_pred'].sum()
    total_actual_goals = df_cal['y_true'].sum()
    print(f"Summe vorhergesagter xG: {total_pred_goals:.2f}  |  Tatsächliche Tore: {total_actual_goals}  "
          f"(Differenz: {total_pred_goals - total_actual_goals:+.2f})")

    return cal_table

def run_xg_experiments(df):
    """
    Durchführen des wissenschaftlichen Vergleichsexperiments (Train-Test-Split 80/20):
    0. Basisquote (konstante Torquote aus Trainingsdaten)
    1. Heuristik (16.5 m-Schwelle, Torquoten aus Trainingsdaten)
    2. Logistische Regression (nur Distanz)
    3. Logistische Regression (Geometrische Features)
    4. Logistische Regression (Erweiterte Features)
    5. Vergleich mit StatsBomb Benchmark
    """
    print("--- STARTE MODELLIERUNG & EVALUATION ---")

    # NaN bereinigen
    df_clean = df.dropna(subset=['distance_to_goal', 'visible_angle', 'is_goal']).copy() # Kopieren der Zeilen in der Distanz, Winkel und Torerfolg ungleich NaN sind

    # Pearson-Korrelation zwischen Distanz und Winkel (vollständiger Datensatz, vor dem Split)
    korrelation = df_clean['distance_to_goal'].corr(df_clean['visible_angle']) # Berechnen der Korrelation
    print(f"\nPearson-Korrelation (distance_to_goal, visible_angle): {korrelation:.4f} (n={len(df_clean)})") # Ausgabe

    # One-Hot-Encoding für kategoriale Features
    body_part_dummies = pd.get_dummies(df_clean['shot_body_part'], prefix='body', drop_first=True) #Nutzen von Dummies für One - Hot - Encoding, mit Prefix "body" für Übersicht, Drop first verhindert redundante Informationen
    play_pattern_dummies = pd.get_dummies(df_clean['play_pattern'], prefix='pattern', drop_first=True)
    
    # Feature-Matrizen definieren
    X_geo = df_clean[['distance_to_goal', 'visible_angle']] # Lernvariable als Konvention immer X
    X_ext = pd.concat([X_geo, body_part_dummies, play_pattern_dummies], axis=1) # Verbinden mithilfe von concat, axis=1 hängt die Dummy-Spalten als zusätzliche Spalten an
    y = df_clean['is_goal'] # Ergebnis bekommt immer Y
    groups = df_clean['match_id'] # Spiel-Zugehörigkeit jedes Schusses für gruppierten Split/Bootstrap

    # Gruppierter Train-Test-Split auf Spielebene (80% Training, 20% Evaluation), Torquote möglichst gleich
    train_idx, test_idx = grouped_train_test_split(y, groups, test_size=0.2, target_rate=y.mean())

    X_geo_train, X_geo_test = X_geo.iloc[train_idx].copy(), X_geo.iloc[test_idx].copy()
    X_ext_train, X_ext_test = X_ext.iloc[train_idx].copy(), X_ext.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]

    # 0. Basisquote: konstante Vorhersage der Torquote aus den Trainingsdaten
    baseline_rate = y_train.mean()
    pred_basisquote = np.full(len(y_test), baseline_rate)
    print(f"Basisquote (konstante Vorhersage aus Trainingsdaten): {baseline_rate*100:.2f}% "
          f"für alle {len(y_test)} Test-Schüsse")

    # 1. Heuristik (16.5 m-Schwelle = Strafraumgrenze)
    # Torquoten je Seite der Schwelle werden ausschließlich aus den Trainingsdaten geschätzt (kein Datenleck)
    near_rate = y_train[X_geo_train['distance_to_goal'] <= 16.5].mean()
    far_rate = y_train[X_geo_train['distance_to_goal'] > 16.5].mean()
    print(f"Heuristik-Torquoten aus Trainingsdaten: <=16.5m: {near_rate*100:.2f}%, "
          f">16.5m: {far_rate*100:.2f}%")

    heuristic_pred = np.where(X_geo_test['distance_to_goal'] <= 16.5, near_rate, far_rate) # Nutzen von WENN DANN SONST von Numpy

    # Skalierung der kontinuierlichen Features (erst NACH der Heuristik, die rohe Meter-Werte braucht)
    # Scaler nur auf Trainingsdaten fitten, Dummy-Spalten (0/1) bleiben unverändert
    continuous_cols = ['distance_to_goal', 'visible_angle']

    scaler_geo = StandardScaler()
    X_geo_train[continuous_cols] = scaler_geo.fit_transform(X_geo_train[continuous_cols])
    print(f"Std (Training) — distance_to_goal: {scaler_geo.scale_[0]:.4f} m, visible_angle: {scaler_geo.scale_[1]:.4f} Grad")
    X_geo_test[continuous_cols] = scaler_geo.transform(X_geo_test[continuous_cols])

    scaler_ext = StandardScaler()
    X_ext_train[continuous_cols] = scaler_ext.fit_transform(X_ext_train[continuous_cols])
    X_ext_test[continuous_cols] = scaler_ext.transform(X_ext_test[continuous_cols])

    # 2. Distanz-Regression: Logistische Regression nur mit der Distanz
    X_dist_train = X_geo_train[['distance_to_goal']]
    X_dist_test = X_geo_test[['distance_to_goal']]

    cv_dist = list(StratifiedGroupKFold(n_splits=5).split(X_dist_train, y_train, groups=train_groups))
    model_dist = LogisticRegressionCV(Cs=10, cv=cv_dist, scoring='neg_log_loss', max_iter=1000, random_state=42,
                                 l1_ratios=(0,), use_legacy_attributes=False)
    model_dist.fit(X_dist_train, y_train)
    report_cv_performance(model_dist, "Distanz-Regression")
    pred_dist = model_dist.predict_proba(X_dist_test)[:, 1]

    # 3. Logistische Regression (Nur Geometrie)
    cv_geo = list(StratifiedGroupKFold(n_splits=5).split(X_geo_train, y_train, groups=train_groups)) # 5 Folds auf Spielebene
    model_geo = LogisticRegressionCV(Cs=10, cv=cv_geo, scoring='neg_log_loss', max_iter=1000, random_state=42,
                                 l1_ratios=(0,), use_legacy_attributes=False) # C wird per CV getuned
    model_geo.fit(X_geo_train, y_train) # Trainieren des Modells
    report_cv_performance(model_geo, "LR Geometrie")
    pred_geo = model_geo.predict_proba(X_geo_test)[:, 1] # Testen des Modells, Wahrscheinlichkeit für kein Tor wird ignoriert

    # 4. Logistische Regression (Erweitert um Körperteile und Spielsituation)
    cv_ext = list(StratifiedGroupKFold(n_splits=5).split(X_ext_train, y_train, groups=train_groups))
    model_ext = LogisticRegressionCV(Cs=10, cv=cv_ext, scoring='neg_log_loss', max_iter=1000, random_state=42,
                                 l1_ratios=(0,), use_legacy_attributes=False)
    model_ext.fit(X_ext_train, y_train)
    report_cv_performance(model_ext, "LR Erweitert")

    # Zusätzliches statsmodels-Modell NUR für die Koeffizienten-Interpretation (unregularisierte ML-Schätzung)
    # Cluster-robuste Standardfehler, geclustert nach match_id (mehrere Schüsse pro Spiel)
    # 'pattern_Other' (4 Schüsse, 0 Tore - im gesamten Datensatz) führt ohne Regularisierung zu quasi-perfekter Separation,
    # daher NUR für die Inferenz in 'pattern_Regular Play' zusammenlegen (X_ext_train/model_ext bleiben unverändert)
    X_ext_train_sm = X_ext_train.astype(float) # Kopie, Dummy-Spalten (bool) in float umwandeln
    X_ext_train_sm['pattern_Regular Play'] = (
        X_ext_train_sm['pattern_Regular Play'] + X_ext_train_sm['pattern_Other']
    )
    X_ext_train_sm = X_ext_train_sm.drop(columns=['pattern_Other'])
    X_ext_train_sm = sm.add_constant(X_ext_train_sm) # Konstante (Intercept) ergänzen

    print(f"\nZusammenlegung für statsmodels: 'pattern_Other' (n={int(X_ext_train['pattern_Other'].sum())} im Training) "
          f"wurde in 'pattern_Regular Play' zusammengeführt (nur für die Inferenz, Vorhersagemodell unverändert).")

    logit_model = sm.Logit(y_train, X_ext_train_sm)
    logit_result = logit_model.fit(cov_type='cluster', cov_kwds={'groups': train_groups})

    print(logit_result.summary())

    coef_table = pd.DataFrame({
        'Koeffizient': logit_result.params,
        'Odds Ratio': np.exp(logit_result.params),
        'KI_unten_95': np.exp(logit_result.conf_int()[0]),
        'KI_oben_95': np.exp(logit_result.conf_int()[1]),
        'p-Wert': logit_result.pvalues,
    })
    print("\nOdds-Ratio-Tabelle mit 95%-Konfidenzintervall (cluster-robust, geclustert nach match_id):")
    print(coef_table.to_string())

    # Einfluss der Features herausfinden
    features = X_ext_train.columns # Die Spaltennamen der Feature-Matrix um Spaltennamen die Koeffizienten zuzuordnen (selbe Reihenfolge in Ausgabe)
    koeffizienten = model_ext.coef_[0] # Die Koeffizienten des Modells sind in 2D Matrix des trainierten Modells, [0] um auf Inhalt zuzugreifen
    odds_ratios = np.exp(koeffizienten) # exp als Umkehrfunktion um die Odds zu erhalten

    coef_reg_df = pd.DataFrame({ # Erstellen eines übersichtlichen DataFrames aus den 3 Spalten
        'Feature': features,
        'Koeffizient (Log-Odds)': koeffizienten,
        'Odds Ratio (Multiplikator)': odds_ratios
    })
    coef_reg_df = coef_reg_df.sort_values(by='Koeffizient (Log-Odds)', ascending=False) # Absteigend nach Koeffizient sortiert (kein Importance-Maß)

    print("\n--- Odds Ratios (LR Erweitert) ---")
    print(coef_reg_df.to_string(index=False)) # Ignorieren der Indizes des DF

    vergleich_df = coef_reg_df.set_index('Feature')[['Koeffizient (Log-Odds)']].join(
        coef_table['Koeffizient'].rename('Koeffizient (unregularisiert)')
    )
    vergleich_df = vergleich_df.rename(columns={'Koeffizient (Log-Odds)': 'Koeffizient (regularisiert)'})
    vergleich_df['Abweichung'] = vergleich_df['Koeffizient (regularisiert)'] - vergleich_df['Koeffizient (unregularisiert)']
    vergleich_df = vergleich_df.dropna()

    print("\n--- VERGLEICH REGULARISIERT VS. UNREGULARISIERT ---")
    print(vergleich_df.to_string())
    print("Hinweis: pattern_Other fehlt hier, da diese Kategorie im unregularisierten Modell "
          "zur Inferenz in pattern_Regular Play zusammengelegt wurde.")

    pred_ext = model_ext.predict_proba(X_ext_test)[:, 1] # Nur Spalte 1 ist relevant - wie wahrscheinlich ist ein Tor

    # Drei exemplarische Schüsse aus dem Testdatensatz extrahieren
    fallstudien_df = df_clean.loc[X_ext_test.index].copy() # Die vollständigen Zeilen der Testschüsse (inkl. Spieler, Körperteil usw.) über den Zeilenindex zurückholen und kopieren
    fallstudien_df['xg_erweitert'] = pred_ext # Neue Spalte mit xG-Wert

    shotmap_df = df_clean.loc[X_ext_test.index, ['x', 'y', 'distance_to_goal', 'visible_angle', 'is_goal']].copy()
    shotmap_df['xg_erweitert'] = pred_ext
    shotmap_df.to_csv("shotmap_export.csv", index=False)
    print("Shot-Map-Daten exportiert: shotmap_export.csv")

    # Szenario A: Distanz-Effekt (Schuss aus großer Distanz, > 25m)
    szenario_a = fallstudien_df[fallstudien_df['distance_to_goal'] > 25].sort_values('distance_to_goal', ascending=False).head(1) # Schüsse mit hoher Distanz absteigend sortiert und den mit der größten Distanz wählen

    # Szenario B: Kopfball-Effekt (Kopfball im Strafraum, Distanz < 16.5 m)
    szenario_b = fallstudien_df[(fallstudien_df['shot_body_part'] == 'Head') & (fallstudien_df['distance_to_goal'] < 16.5)].head(1) # Kopfball und Distanz unter 16.5 meter

    # Szenario C: Konter-Bonus (kein Kopfball im Strafraum nach Konter, aus brauchbarem Winkel)
    szenario_c = fallstudien_df[
        (fallstudien_df['play_pattern'] == 'From Counter') & # Spielzug ist zwingend ein Konter
        (fallstudien_df['shot_body_part'] != 'Head') & # Zwingend kein Kopfball
        (fallstudien_df['distance_to_goal'] < 16.5) & # Abschluss im Strafraum
        (fallstudien_df['visible_angle'] >= 15) # Kein Schuss aus spitzem Winkel nahe der Grundlinie
    ].head(1) # Wähle den Ersten

    spalten = ['match_id', 'player', 'distance_to_goal', 'visible_angle', 'shot_body_part', 'play_pattern', 'is_goal', 'xg_erweitert']

    print("\n--- SZENARIO A: DISTANZ-EFFEKT ---")
    print(szenario_a[spalten].to_string(index=False))

    print("\n--- SZENARIO B: KOPFBALL-EFFEKT ---")
    print(szenario_b[spalten].to_string(index=False))

    print("\n--- SZENARIO C: KONTER-BONUS ---")
    print(szenario_c[spalten].to_string(index=False))

    # 5. Benchmark: StatsBomb xG Modell
    missing_xg_count = df_clean.loc[y_test.index, 'statsbomb_xg'].isna().sum() # Überprüfen ob das Ergebnis durch NaN - Werte im sb-modell verfälscht wird 
    assert missing_xg_count == 0, "StatsBomb-xG fehlt für Testschüsse"
    statsbomb_test = df_clean.loc[y_test.index, 'statsbomb_xg'] # Extrahieren aller statsbomb_xg - Werte der relevanten Zeilen (die Zeilen die wir zum Testen nutzen)

    # Gruppierter Bootstrap auf Spielebene für 95%-Konfidenzintervalle der Testmetriken
    test_groups = groups.iloc[test_idx]
    print("\n--- BOOTSTRAP 95%-KONFIDENZINTERVALLE (gruppierter Bootstrap, n=1000) ---")
    modelle = [ # Modell-Kette 0-5 als (Name, Vorhersage) - gemeinsam für Bootstrap und Ergebnistabelle
        ("0. Basisquote", pred_basisquote),
        ("1. Heuristik (16.5 m)", heuristic_pred),
        ("2. Distanz-Regression", pred_dist),
        ("3. LR (Geometrie)", pred_geo),
        ("4. LR (Erweitert)", pred_ext),
        ("5. StatsBomb (Benchmark)", statsbomb_test),
    ]
    for name, scores in modelle:
        print(f"\n{name}:")
        bootstrap_ci(y_test, scores, test_groups, n_boot=1000)

    # Nutzen von scikit um Test und anschließend Vergleich durchzuführen (als Dictionary)
    results = {
        'Modell': [name for name, _ in modelle],
        'Log-Loss (niedriger=besser)': [log_loss(y_test, scores) for _, scores in modelle], # Log - Loss Vergleich - echtes Ergebnis, vorhergesagtes Ergebnis
        'ROC-AUC (hoeher=besser)': [roc_auc_score(y_test, scores) for _, scores in modelle],
        'Brier Score (niedriger=besser)': [brier_score_loss(y_test, scores) for _, scores in modelle]
    }

    results_df = pd.DataFrame(results) # Erstellen eines df aus dem Ergebnis - Dictionary
    print("\nEVALUATIONSERGEBNISSE (TEST-DATENSET):")
    print(results_df.to_string(index=False)) # Formatierung des df für die Ausgabe in der Konsole

    # Kalibrierungsprüfung (Quintile wegen kleinem Testdatensatz) für LR Erweitert und StatsBomb
    print("\n--- KALIBRIERUNGSPRÜFUNG ---")
    calibration_check(y_test, pred_ext, "LR (Erweitert)")
    calibration_check(y_test, statsbomb_test, "StatsBomb (Benchmark)")

 
    return results_df

if __name__ == "__main__":
    print("\n==========================================")
    print("   xG-MODELL PIPELINE (WM 2022)   ")
    print("==========================================\n")

 # !Wichtig! Sollte calculate_distance/calculate_angle geändert werden, muss auch die csv gelöscht und neu geladen werden
    # Phase 1: Lade komplettes Turnier (WM 2022: competition_id=43, season_id=106)
    if os.path.exists("shots_wm2022.csv"): # Durchsuchen der Festplatte um ständiges herunterladen zu verhindern
        print("Nutze Cache: shots_wm2022.csv")
        dataset = pd.read_csv("shots_wm2022.csv")
        if os.path.exists("shots_wm2022_meta.json"): # Beim Download gespeicherte Kennzahlen ausgeben
            with open("shots_wm2022_meta.json", encoding="utf-8") as f:
                print_extraction_stats(json.load(f))
        else:
            print("Hinweis: shots_wm2022_meta.json fehlt - für die Extraktions-Kennzahlen CSV löschen und neu laden.")
    else:
        dataset, extraction_stats = fetch_season_shots(competition_id=43, season_id=106) # Sollte die CSV nicht gefunden werden, Daten herunterladen und in CSV schreiben
        dataset.to_csv("shots_wm2022.csv", index=False)
        with open("shots_wm2022_meta.json", "w", encoding="utf-8") as f: # Kennzahlen zusammen mit der CSV speichern
            json.dump(extraction_stats, f, indent=2, ensure_ascii=False)
        print()
        print_extraction_stats(extraction_stats)

    # Phase 2: Qualitätscheck durchführen 
    verify_data_quality(dataset)

    # Phase 3 & 4: Vergleichsexperiment ausführen und Metriken berechnen
    eval_table = run_xg_experiments(dataset)
