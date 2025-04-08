#!/bin/zsh

# Chemins de base
CONFIG_DIR="realworld_results/generated_configs"
SAVEBASE="realworld_results"
MATFILE="LVZ/output_ind_lin_1178.mat"
EXPLORATION_DIR="$HOME/Projets/RECHERCHES/Recherches/MIGRATE/reps/Migrate/exploration/bayhunter/MCMC_RealWorld_Inversion"

mkdir -p "$EXPLORATION_DIR"

# Boucle sur les fichiers ini
for config_file in ${CONFIG_DIR}/config_*.ini; do
    config_name=$(basename $config_file)
    suffix=${config_name:r}
    suffix_number=${suffix#config_}

    test_dir="${SAVEBASE}/test_${suffix_number}"
    output_dir="${EXPLORATION_DIR}/test_${suffix_number}"

    echo ">> Running simulation for test_${suffix_number}"

    # Lire le nombre de chaînes depuis le fichier ini
    nchains=$(grep "^nchains" "$config_file" | awk -F '=' '{gsub(/ /, "", $2); print $2}')
    if [[ -z "$nchains" ]]; then
        echo "Erreur: impossible de lire 'nchains' depuis $config_file"
        continue
    fi

    # Lancer l'inversion
    python3 realworld_vs_simulators.py \
        --mat-file "$MATFILE" \
        --savepath "$test_dir" \
        --ini-file "$config_file"

    # Créer le répertoire de destination
    mkdir -p "$output_dir"

    # Extraire les meilleures courbes
    python3 best_dispersion_curves_per_chains.py \
        --mat-file "$MATFILE" \
        --directory "${test_dir}/data" \
        --n-chains "$nchains" \
        --output-directory "$output_dir"

    # Copier le fichier ini pour trace
    cp "$config_file" "$output_dir/"
done
