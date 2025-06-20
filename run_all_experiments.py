import os
import sys
import glob
import time
from multiprocessing import Pool, cpu_count
import traceback

# Załóżmy, że ExperimentObjects.Experiment jest importowalne
from ExperimentObjects.Experiment import Experiment


def is_experiment_completed(config_file_path: str, base_results_dir: str) -> bool:
    """
    Sprawdza czy eksperyment dla danej konfiguracji został już ukończony.
    Zwraca True jeśli eksperyment został ukończony, False w przeciwnym razie.
    """
    config_name = os.path.splitext(os.path.basename(config_file_path))[0]
    exp_dir = os.path.join(base_results_dir, config_name)

    # Sprawdź czy folder eksperymentu istnieje
    if not os.path.isdir(exp_dir):
        return False

    # Sprawdź czy istnieją kluczowe pliki wynikowe
    genome_file = os.path.join(exp_dir, 'best_genome.pkl')
    states_file = os.path.join(exp_dir, 'game_states.json')

    return os.path.isfile(genome_file) and os.path.isfile(states_file)


def run_experiment_for_config(
        config_file_path: str,
        base_results_dir: str,
        generations_count: int,
) -> tuple[str, float | None, str | None]:
    config_name = os.path.splitext(os.path.basename(config_file_path))[0]

    # Sprawdź czy eksperyment już został ukończony
    if is_experiment_completed(config_file_path, base_results_dir):
        print(f"Eksperyment dla konfiguracji {config_name} już został ukończony. Pomijam...")
        try:
            # Załaduj istniejące wyniki
            exp = Experiment(
                config_path=config_file_path,
                output_dir=base_results_dir,
                generations=generations_count,
            )
            final_fitness = exp.load_results()
            return config_name, final_fitness, exp.states_path
        except Exception as e:
            print(f"Błąd podczas ładowania wyników dla {config_name}: {e}")
            print("Eksperyment zostanie uruchomiony ponownie...")

    print(f"Rozpoczynanie eksperymentu dla konfiguracji: {config_name}...")
    try:
        exp = Experiment(
            config_path=config_file_path,
            output_dir=base_results_dir,
            generations=generations_count,
        )
        final_fitness = exp.run()
        print(
            f"Zakończono eksperyment dla {config_name}. Końcowy fitness: {final_fitness:.2f}"
        )
        return config_name, final_fitness, exp.states_path
    except FileNotFoundError as fnfe:
        print(
            f"BŁĄD KRYTYCZNY (FileNotFoundError) podczas przetwarzania konfiguracji {config_name}: {fnfe}"
        )
        print(f"  Sprawdzana ścieżka do pliku konfiguracyjnego: {config_file_path}")
        traceback.print_exc()
        return config_name, None, None
    except KeyError as ke:
        print(
            f"BŁĄD KRYTYCZNY (KeyError) podczas przetwarzania konfiguracji {config_name}: {ke}"
        )
        print(
            f"  Sprawdź, czy plik konfiguracyjny {config_file_path} zawiera wszystkie wymagane sekcje/klucze."
        )
        traceback.print_exc()
        return config_name, None, None
    except Exception as e:
        print(
            f"Inny błąd podczas przetwarzania konfiguracji {config_name}: {type(e).__name__} - {e}"
        )
        traceback.print_exc()
        return config_name, None, None


def get_experiment_progress(config_files_list: list, base_results_dir: str) -> tuple[list, list]:
    """
    Sprawdza postęp eksperymentów i zwraca listy ukończonych i pozostałych do wykonania.
    """
    completed_configs = []
    remaining_configs = []

    for config_path in config_files_list:
        if is_experiment_completed(config_path, base_results_dir):
            completed_configs.append(config_path)
        else:
            remaining_configs.append(config_path)

    return completed_configs, remaining_configs


# --- Główna część skryptu ---
if __name__ == "__main__":
    # Ustawienia
    configs_folder_name = "Configs"
    results_main_dir = "experiment_results_parallel"
    num_generations_per_exp = 1000  # Zmieniono z 225 na 1000

    os.makedirs(results_main_dir, exist_ok=True)

    # --- OKREŚLANIE ŚCIEŻEK (Z OBSŁUGĄ BRAKU __file__) ---
    try:
        # Preferowana metoda, jeśli uruchamiane jako skrypt .py
        current_script_directory = os.path.dirname(os.path.abspath(__file__))
        project_root_dir = current_script_directory
        print(f"Użyto __file__ do określenia project_root_dir: {project_root_dir}")
    except NameError:
        # Fallback dla środowisk interaktywnych (np. Jupyter Notebook)
        project_root_dir = os.getcwd()
        print(
            f"OSTRZEŻENIE: __file__ nie jest zdefiniowane. Użyto os.getcwd() jako project_root_dir: {project_root_dir}"
        )
        print(
            "  Upewnij się, że uruchamiasz ten kod z głównego katalogu projektu."
        )

    absolute_configs_dir = os.path.join(project_root_dir, configs_folder_name)
    # --- KONIEC OKREŚLANIA ŚCIEŻEK ---

    config_files_relative_or_absolute = glob.glob(
        os.path.join(absolute_configs_dir, "*.ini")
    )
    if not config_files_relative_or_absolute:
        print(
            f"Nie znaleziono plików konfiguracyjnych w folderze: {absolute_configs_dir}"
        )
        print(f"  Sprawdzany katalog główny projektu: {project_root_dir}")
        print(f"  Bieżący katalog roboczy (CWD) podczas uruchamiania: {os.getcwd()}")
        sys.exit(1)

    config_files_list = [
        os.path.abspath(cfp) for cfp in config_files_relative_or_absolute
    ]

    print(f"Znaleziono {len(config_files_list)} plików konfiguracyjnych.")

    # Sprawdź postęp eksperymentów
    completed_configs, remaining_configs = get_experiment_progress(config_files_list, results_main_dir)

    print(f"Ukończone eksperymenty: {len(completed_configs)}")
    print(f"Pozostałe eksperymenty do wykonania: {len(remaining_configs)}")

    if completed_configs:
        print("Przykłady ukończonych eksperymentów:")
        for i, cfp in enumerate(completed_configs[:3]):
            config_name = os.path.splitext(os.path.basename(cfp))[0]
            print(f"  {i + 1}. {config_name}")

    if not remaining_configs:
        print("Wszystkie eksperymenty zostały już ukończone!")

        # Wczytaj wyniki ukończonych eksperymentów
        print("\n--- Ładowanie wyników ukończonych eksperymentów ---")
        tasks = [
            (cfp, results_main_dir, num_generations_per_exp)
            for cfp in completed_configs
        ]

        with Pool(processes=min(len(completed_configs), cpu_count())) as pool:
            results_data = pool.starmap(run_experiment_for_config, tasks)
    else:
        print(f"\nRozpoczynam przetwarzanie {len(remaining_configs)} pozostałych eksperymentów...")

        tasks = [
            (cfp, results_main_dir, num_generations_per_exp)
            for cfp in remaining_configs
        ]

        num_processes_to_use = min(len(remaining_configs), cpu_count())
        print(
            f"Uruchamianie {len(tasks)} eksperymentów równolegle przy użyciu {num_processes_to_use} procesów..."
        )

        start_timestamp = time.time()

        with Pool(processes=num_processes_to_use) as pool:
            remaining_results = pool.starmap(run_experiment_for_config, tasks)

        end_timestamp = time.time()
        print(f"\n--- Pozostałe eksperymenty zakończone ---")
        print(
            f"Czas przetwarzania pozostałych eksperymentów: {end_timestamp - start_timestamp:.2f} sekund."
        )

        # Wczytaj również wyniki już ukończonych eksperymentów
        print("\n--- Ładowanie wyników wcześniej ukończonych eksperymentów ---")
        completed_tasks = [
            (cfp, results_main_dir, num_generations_per_exp)
            for cfp in completed_configs
        ]

        if completed_tasks:
            with Pool(processes=min(len(completed_configs), cpu_count())) as pool:
                completed_results = pool.starmap(run_experiment_for_config, completed_tasks)
            results_data = remaining_results + completed_results
        else:
            results_data = remaining_results

    print(f"\n--- Wszystkie eksperymenty przetworzone ---")

    print("\n--- Podsumowanie wyników ---")
    successful_exp_results = []
    for name, fitness_val, states_file_path in results_data:
        if fitness_val is not None:
            print(
                f"Konfiguracja: {name}, Końcowy Fitness: {fitness_val:.2f}, Stany gry: {states_file_path}"
            )
            successful_exp_results.append((name, fitness_val, states_file_path))
        else:
            print(f"Konfiguracja: {name} - nie powiodła się lub wystąpił błąd.")

    if successful_exp_results:
        successful_exp_results.sort(key=lambda x: x[1], reverse=True)

    print("\n--- Najlepsze konfiguracje (Top 10) ---")
    if not successful_exp_results:
        print("Brak udanych eksperymentów do wyświetlenia.")
    else:
        for i, (name, fitness_val, _) in enumerate(
                successful_exp_results[:10]  # Zwiększono z 5 do 10
        ):
            print(f"{i + 1}. {name}: Fitness = {fitness_val:.2f}")

    if successful_exp_results:
        best_cfg_name, best_fitness, best_states_file = successful_exp_results[0]
        print(f"\n--- Najlepszy eksperyment: {best_cfg_name} (Fitness: {best_fitness:.2f}) ---")

        original_config_file_for_replay = None
        for cfp_abs in config_files_list:
            if os.path.splitext(os.path.basename(cfp_abs))[0] == best_cfg_name:
                original_config_file_for_replay = cfp_abs
                break

        if original_config_file_for_replay:
            print(f"Oryginalny plik konfiguracyjny: {original_config_file_for_replay}")
            print(f"Plik stanów gry: {best_states_file}")
            print(f"\nAby odtworzyć najlepszy eksperyment:")
            print(
                f"  exp_replayer = Experiment(config_path='{original_config_file_for_replay}', output_dir='{results_main_dir}')")
            print(f"  exp_replayer.replay()")

            # Opcjonalnie: automatyczne odtworzenie
            user_input = input("\nCzy chcesz automatycznie odtworzyć najlepszy eksperyment? (y/n): ")
            if user_input.lower() in ['y', 'yes', 't', 'tak']:
                try:
                    print(f"\nAutomatyczne odtwarzanie najlepszego wyniku dla: {best_cfg_name}...")
                    exp_replayer_auto = Experiment(
                        config_path=original_config_file_for_replay,
                        output_dir=results_main_dir,
                    )
                    if best_states_file and os.path.exists(best_states_file):
                        exp_replayer_auto.replay(delay=0.05)
                    else:
                        print(f"Nie można znaleźć pliku stanów gry: {best_states_file}")
                except Exception as e:
                    print(f"Błąd podczas automatycznego odtworzenia: {e}")
                    traceback.print_exc()
        else:
            print(f"Nie udało się znaleźć oryginalnego pliku .ini dla '{best_cfg_name}'")
    else:
        print("\nBrak udanych eksperymentów, więc nie ma czego odtwarzać.")

    print("\nGotowe.")