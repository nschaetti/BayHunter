import configparser
import os

# Dossier de sortie
output_dir = "realworld_results/generated_configs"
os.makedirs(output_dir, exist_ok=True)

# Grille de paramètres à explorer
vpvs_values = [(1.4, 2.1), (1.6, 2.0)]
vs_ranges = [(2.0, 5.0), (2.3, 4.2)]
layer_ranges = [(1, 20), (3, 25)]
lvz_values = [None, 0.05, 0.1]
thickmin_values = [0.05, 0.1]
nchains_values = [5, 10]
acceptance_ranges = [(40, 45), (50, 55)]
mohoest_values = [None, (30, 2), (35, 3)]
propdist_values = [
    '0.015, 0.015, 0.015, 0.005, 0.005',  # valeurs standard
    '0.01, 0.01, 0.01, 0.01, 0.01',       # plus conservateur
    '0.02, 0.02, 0.02, 0.01, 0.01'        # plus agressif
]

# Génération des fichiers ini
idx = 0
for vpvs in vpvs_values:
    for vs in vs_ranges:
        for layers in layer_ranges:
            for lvz in lvz_values:
                for thickmin in thickmin_values:
                    for nchains in nchains_values:
                        for acc in acceptance_ranges:
                            for mohoest in mohoest_values:
                                for propdist in propdist_values:
                                    config = configparser.ConfigParser()

                                    config['modelpriors'] = {
                                        'vpvs': f'{vpvs[0]}, {vpvs[1]}',
                                        'layers': f'{layers[0]}, {layers[1]}',
                                        'vs': f'{vs[0]}, {vs[1]}',
                                        'z': '0, 60',
                                        'mohoest': 'None' if mohoest is None else f'{mohoest[0]}, {mohoest[1]}',
                                        'rfnoise_corr': '0.9',
                                        'swdnoise_corr': '0.',
                                        'rfnoise_sigma': '1e-5, 0.05',
                                        'swdnoise_sigma': '1e-5, 0.05'
                                    }

                                    config['initparams'] = {
                                        'nchains': str(nchains),
                                        'iter_burnin': str(2048 * 16),
                                        'iter_main': str(2048 * 32),
                                        'propdist': propdist,
                                        'acceptance': f'{acc[0]}, {acc[1]}',
                                        'thickmin': str(thickmin),
                                        'lvz': 'None' if lvz is None else str(lvz),
                                        'hvz': 'None',
                                        'rcond': '1e-5',
                                        'station': f'test_{idx}',
                                        'savepath': f'realworld_results/test_{idx}',
                                        'maxmodels': '50000'
                                    }

                                    filename = f'config_{idx:03d}.ini'
                                    filepath = os.path.join(output_dir, filename)
                                    with open(filepath, 'w') as configfile:
                                        config.write(configfile)

                                    idx += 1
