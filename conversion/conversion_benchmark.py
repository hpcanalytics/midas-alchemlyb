import pandas as pd
import numpy as np
import time
import logging, sys
logging.basicConfig(stream=sys.stdout,level=logging.INFO)

def create_df_by_size(mb_by_factor_of_two):
    size = mb_by_factor_of_two
    df = pd.DataFrame(np.random.randint(0, 2**10*(2**size),
        size=(2**10*(2**size), 2**6)), columns=[ 'col' + str(x) for x in
            range(2**6)]) 
    return df

def _convert_to_hdf(df, fname):
    df.to_hdf("{}.h5".format(fname), key='df', mode='w')

def _convert_to_csv(df, fname):
    df.to_csv("{}.csv".format(fname), mode='w')

def _convert_to_parquet(df, fname, engine='auto', compression='snappy'):
    print ("{},{}".format(engine, compression))
    df.to_parquet("{}.parquet.{}.{}".format(fname, engine, compression), engine=engine, compression=compression)

def convert(df, ctype="hdf", report=True):
    fname = df.memory_usage().sum()
    s0 = time.time()
    if ctype[:7] == "parquet":
        ctype, engine, compression = ctype.split("_")
        globals()["_convert_to_{}".format(ctype)](df, fname, engine, compression)
    else:
        globals()["_convert_to_{}".format(ctype)](df, fname)
    s1 = time.time()
    logging.info("{},{},{},{},{}".format(fname, ctype, s1-s0, s1, s0))

if __name__ == "__main__":
    ctypes = ["hdf", "csv"]
    parquets = []
    for engine in ['pyarrow' ]:#, 'fastparquet']:
        for compression in ['snappy', 'gzip', 'brotli', 'None']:
            parquets.append("parquet_{}_{}".format(engine, compression))
    ctypes += parquets
    for size in range(1,2):
        df = create_df_by_size(size)
        for ctype in ctypes:
            # Exception
            # fastparquet,snappy
            # Assertion failed: (PassInf && "Expected all immutable passes to be initialized"), function addImmutablePass, file /Users/buildbot/miniconda3/conda-bld/llvmdev_1545076115094/work/lib/IR/LegacyPassManager.cpp, line 812.
            # Abort trap: 6
            convert(df, ctype)
