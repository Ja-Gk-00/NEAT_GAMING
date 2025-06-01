import os
import sys
import glob
import time
from multiprocessing import Pool, cpu_count
import traceback

# Załóżmy, że ExperimentObjects.Experiment jest importowalne
# (jak w poprzedniej wersji)
from ExperimentObjects.Experiment import Experiment


# --- Funkcja do uruchamiania pojedynczego eksperymentu (bez zmian) ---
def run_experiment_for_config(
    config_file_path: str,
    base_results_dir: str,
    generations_count: int,
) -> tuple[str, float | None, str | None]:
    config_name = os.path.splitext(os.path.basename(config_file_path))[0]
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


# --- Główna część skryptu ---
if __name__ == "__main__":
    # Ustawienia
    configs_folder_name = "Configs"
    results_main_dir = "experiment_results_parallel"
    num_generations_per_exp = 225

    os.makedirs(results_main_dir, exist_ok=True)

    # --- OKREŚLANIE ŚCIEŻEK (Z OBSŁUGĄ BRAKU __file__) ---
    try:
        # Preferowana metoda, jeśli uruchamiane jako skrypt .py
        current_script_directory = os.path.dirname(os.path.abspath(__file__))
        # Zakładamy, że skrypt jest w głównym katalogu projektu
        project_root_dir = current_script_directory
        # Jeśli skrypt byłby w podkatalogu (np. NEAT_GAMING/Scripts/):
        # project_root_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
        print(f"Użyto __file__ do określenia project_root_dir: {project_root_dir}")
    except NameError:
        # Fallback dla środowisk interaktywnych (np. Jupyter Notebook)
        # Zakłada, że bieżący katalog roboczy to katalog główny projektu.
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

    print(
        f"Znaleziono {len(config_files_list)} plików konfiguracyjnych do przetworzenia."
    )
    for i, cfp in enumerate(config_files_list[:3]):
        print(f"  Przykład pliku konfiguracyjnego {i+1}: {cfp}")

    tasks = [
        (cfp, results_main_dir, num_generations_per_exp)
        for cfp in config_files_list
    ]

    num_processes_to_use = min(len(config_files_list), cpu_count())
    print(
        f"Uruchamianie {len(tasks)} eksperymentów równolegle przy użyciu {num_processes_to_use} procesów..."
    )

    start_timestamp = time.time()

    with Pool(processes=num_processes_to_use) as pool:
        results_data = pool.starmap(run_experiment_for_config, tasks)

    end_timestamp = time.time()
    print(f"\n--- Wszystkie eksperymenty zakończone ---")
    print(
        f"Całkowity czas przetwarzania: {end_timestamp - start_timestamp:.2f} sekund."
    )

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

    print("\n--- Najlepsze konfiguracje (Top 5) ---")
    if not successful_exp_results:
        print("Brak udanych eksperymentów do wyświetlenia.")
    else:
        for i, (name, fitness_val, _) in enumerate(
            successful_exp_results[:5]
        ):
            print(f"{i+1}. {name}: Fitness = {fitness_val:.2f}")

    if successful_exp_results:
        best_cfg_name, _, best_states_file = successful_exp_results[0]
        print(f"\n--- Odtwarzanie najlepszego eksperymentu ({best_cfg_name}) ---")

        original_config_file_for_replay = None
        for cfp_abs in config_files_list:
            if os.path.splitext(os.path.basename(cfp_abs))[0] == best_cfg_name:
                original_config_file_for_replay = cfp_abs
                break

        if original_config_file_for_replay:
            print(
                f"Oryginalny plik konfiguracyjny: {original_config_file_for_replay}"
            )
            print(f"Plik stanów gry: {best_states_file}")
            print(f"\nAby odtworzyć ręcznie (przykład):")
            print(
                f"  exp_replayer = Experiment(config_path='{original_config_file_for_replay}', output_dir='{results_main_dir}')"
            )
            print(f"  if '{best_states_file}' and os.path.exists('{best_states_file}'):")
            print(
                f"      exp_replayer.game_play.replay('{best_states_file}', delay=0.05)"
            )
            print(f"  else: print('Plik stanów gry {best_states_file} nie istnieje lub jest pusty.')")

            try:
                print(
                    f"\nAutomatyczne odtwarzanie najlepszego wyniku dla: {best_cfg_name}..."
                )
                exp_replayer_auto = Experiment(
                    config_path=original_config_file_for_replay,
                    output_dir=results_main_dir,
                )
                if best_states_file and os.path.exists(best_states_file):
                    exp_replayer_auto.game_play.replay(best_states_file, delay=0.05)
                else:
                    print(
                        f"Nie można znaleźć pliku stanów gry do automatycznego odtworzenia: {best_states_file}"
                    )
            except Exception as e:
                print(f"Błąd podczas próby automatycznego odtworzenia: {e}")
                traceback.print_exc()
        else:
            print(
                f"Nie udało się znaleźć oryginalnego pliku .ini dla najlepszej konfiguracji '{best_cfg_name}' do odtworzenia."
            )
    else:
        print("\nBrak udanych eksperymentów, więc nie ma czego odtwarzać.")

    print("\nGotowe.")

