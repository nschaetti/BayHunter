#
# MIGRATE
#

# Imports
from typing import List
import pyarrow as pa
import pyarrow.parquet as pq
from BayHunter.data import SeismicSample


def save_samples_to_arrow(samples: List[SeismicSample], output_path: str):
    """
    Save a list of SeismicSample objects to an Arrow file.
    :param samples: List of SeismicSample objects
    :param output_path: Path to the output Arrow file
    """
    # Transform
    dicts = [s.to_arrow_dict() for s in samples]

    # Regroupe par colonnes
    table = pa.table({k: [d[k] for d in dicts] for k in dicts[0]})

    # Écrit
    pq.write_table(table, output_path, compression='zstd')
# end save_samples_to_arrow

