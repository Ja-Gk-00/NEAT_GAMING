import os
import sys
import traceback

# --- Konfiguracja ścieżek i importy ---
# Upewnij się, że ExperimentObjects.Experiment jest importowalne.
# Jeśli ten skrypt jest w głównym katalogu projektu (NEAT_GAMING),
# a ExperimentObjects to podkatalog, import powinien działać.
try:
    # Preferowana metoda, jeśli uruchamiane jako skrypt .py
    SCRIPT_DIR_FOR_IMPORT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback dla środowisk interaktywnych (np. Jupyter Notebook)
    # Zakłada, że bieżący katalog roboczy to katalog główny projektu.
    SCRIPT_DIR_FOR_IMPORT = os.getcwd()
    print(
        f"OSTRZEŻENIE: __file__ nie jest zdefiniowane. Użyto os.getcwd() jako bazę dla importów: {SCRIPT_DIR_FOR_IMPORT}"
    )

# Dodaj katalog główny projektu do sys.path, jeśli go tam nie ma,
# aby umożliwić import ExperimentObjects
if SCRIPT_DIR_FOR_IMPORT not in sys.path:
    sys.path.insert(0, SCRIPT_DIR_FOR_IMPORT)

try:
    from ExperimentObjects.Experiment import Experiment
except ImportError as e:
    print(f"Błąd importu klasy Experiment: {e}")
    print("Upewnij się, że skrypt jest uruchamiany z głównego katalogu projektu NEAT_GAMING,")
    print("lub że ścieżka do ExperimentObjects jest poprawnie dodana do sys.path.")
    sys.exit(1)

# --- Stałe konfiguracyjne dla skryptu ---
RESULTS_BASE_DIR_NAME = "experiment_results_parallel" # Nazwa głównego folderu z wynikami
ORIGINAL_CONFIGS_DIR_NAME = "Configs" # Nazwa folderu z oryginalnymi plikami .ini
STATES_FILENAME = "game_states.json" # Nazwa pliku ze stanami gry
REPLAY_DELAY = 0.07 # Opóźnienie między klatkami powtórki (w sekundach)

def replay_all_experiments_sequentially():
    """
    Iteruje przez wszystkie podfoldery w katalogu wyników i odtwarza
    powtórki, czekając na akcję użytkownika między każdą z nich.
    """
    try:
        project_root_dir = SCRIPT_DIR_FOR_IMPORT # Używamy wcześniej zdefiniowanej ścieżki
    except NameError: # Powinno być już obsłużone, ale dla pewności
        project_root_dir = os.getcwd()
        print(f"Używam os.getcwd() jako project_root_dir: {project_root_dir}")


    results_base_path = os.path.join(project_root_dir, RESULTS_BASE_DIR_NAME)
    original_configs_path = os.path.join(project_root_dir, ORIGINAL_CONFIGS_DIR_NAME)

    if not os.path.isdir(results_base_path):
        print(f"BŁĄD: Katalog wyników '{results_base_path}' nie istnieje.")
        return

    if not os.path.isdir(original_configs_path):
        print(f"BŁĄD: Katalog oryginalnych konfiguracji '{original_configs_path}' nie istnieje.")
        return

    # Pobierz listę wszystkich podfolderów (eksperymentów) w katalogu wyników
    try:
        experiment_folders = sorted([
            d for d in os.listdir(results_base_path)
            if os.path.isdir(os.path.join(results_base_path, d))
        ])
    except OSError as e:
        print(f"Błąd podczas listowania katalogu wyników '{results_base_path}': {e}")
        return

    if not experiment_folders:
        print(f"Nie znaleziono żadnych folderów eksperymentów w '{results_base_path}'.")
        return

    print(f"Znaleziono {len(experiment_folders)} folderów eksperymentów do przejrzenia.")
    print("--- Rozpoczynanie sekwencyjnego odtwarzania ---")

    for i, exp_folder_name in enumerate(experiment_folders):
        print(f"\n--- Eksperyment {i+1}/{len(experiment_folders)}: {exp_folder_name} ---")

        current_exp_dir = os.path.join(results_base_path, exp_folder_name)
        states_file_path = os.path.join(current_exp_dir, STATES_FILENAME)

        # Nazwa pliku .ini jest taka sama jak nazwa folderu eksperymentu
        original_ini_filename = f"{exp_folder_name}.ini"
        original_ini_path = os.path.join(original_configs_path, original_ini_filename)

        if not os.path.isfile(states_file_path):
            print(f"  Pominięto: Plik stanów gry '{STATES_FILENAME}' nie znaleziony w '{current_exp_dir}'.")
            continue

        if not os.path.isfile(original_ini_path):
            print(f"  Pominięto: Oryginalny plik konfiguracyjny '{original_ini_filename}' nie znaleziony w '{original_configs_path}'.")
            continue

        print(f"  Odtwarzanie z pliku stanów: {states_file_path}")
        print(f"  Używana konfiguracja: {original_ini_path}")

        try:
            # Tworzymy obiekt Experiment.
            # output_dir dla konstruktora Experiment to katalog, w którym Experiment
            # spodziewa się znaleźć/stworzyć podfolder o nazwie exp_folder_name.
            # W tym przypadku jest to results_base_path.
            exp_replayer = Experiment(
                config_path=original_ini_path,
                output_dir=results_base_path
            )
            exp_replayer.game_play.replay(states_file_path, delay=REPLAY_DELAY)
        except Exception as e:
            print(f"  Błąd podczas odtwarzania eksperymentu '{exp_folder_name}': {e}")
            traceback.print_exc()
            # Kontynuuj do następnego, nawet jeśli jeden się nie powiedzie

        if i < len(experiment_folders) - 1: # Jeśli to nie jest ostatni eksperyment
            user_action = input("Naciśnij Enter, aby przejść do następnej powtórki, lub 'q' aby zakończyć: ")
            if user_action.lower() == 'q':
                print("Zakończono przeglądanie powtórek na życzenie użytkownika.")
                break
        else:
            print("\nTo była ostatnia powtórka.")

    print("\n--- Zakończono sekwencyjne odtwarzanie wszystkich dostępnych eksperymentów ---")

if __name__ == "__main__":
    replay_all_experiments_sequentially()
