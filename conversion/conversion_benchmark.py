import pandas as pd
import numpy as np
import time
import os
import logging, sys
logging.basicConfig(stream=sys.stdout,level=logging.INFO)

def create_df_by_size(mb_by_factor_of_two):
    size = mb_by_factor_of_two
    df = pd.DataFrame(np.random.randint(0, 2**10*(2**size),
        size=(2**10*(2**size), 2**6)), columns=[ 'col' + str(x) for x in
            range(2**6)]) 
    return df

def _convert_to_hdf(df, fname, key="df"):
    df.to_hdf("{}.h5".format(fname), key=key, mode='w')

def _convert_to_csv(df, fname):
    df.to_csv("{}.csv".format(fname), mode='w')

def _convert_to_parquet(df, fname, engine='auto', compression='snappy'):
    #print ("{},{}".format(engine, compression))
    df.to_parquet("{}.parquet.{}.{}".format(fname, engine, compression), engine=engine, compression=compression)

def _convert_to_pickle(df, fname):
    df.to_pickle("{}.pkl".format(fname))

def _convert_to_sql(df, fname):
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///{}.db'.format(fname), echo=False)
    df.to_sql(str(fname), con=engine, if_exists='append')

# openpyxl
def _convert_to_excel(df, fname):
    df.to_excel("{}.xlsx".format(fname))

def _convert_to_json(df, fname):
    df.to_json("{}.json".format(fname))

def _convert_to_feather(df, fname):
    df.to_feather("{}.feather".format(fname))

def _convert_to_stata(df, fname):
    df.to_stata("{}.dta".format(fname))

def _convert_to_msgpack(df, fname):
    df.to_msgpack("{}.msgpack".format(fname))

def convert(df, ctype="hdf"):
    fname = df.memory_usage().sum()
    s0 = time.time()
    if ctype[:7] == "parquet":
        nctype, engine, compression = ctype.split("_")
        if compression == "None":
            compression = None
        globals()["_convert_to_{}".format(nctype)](df, fname, engine, compression)
    else:
        globals()["_convert_to_{}".format(ctype)](df, fname)
    s1 = time.time()
    logging.info("convert,{},{},{},{},{}".format(fname, ctype, s1-s0, s1, s0))
    return fname

def _load_from_hdf(fname):
    return pd.read_hdf("{}.h5".format(fname), 'df')

def _load_from_csv(fname):
    return pd.read_csv("{}.csv".format(fname))

def _load_from_parquet(fname, engine, compression):
    return pd.read_parquet("{}.parquet.{}.{}".format(fname, engine, compression))

def _load_from_pickle(fname):
    return pd.read_pickle("{}.pkl".format(fname))

def _load_from_sql(fname):
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///{}.db'.format(fname), echo=False)
    return pd.read_sql_table(str(fname), con=engine)

def _load_from_json(fname):
    return pd.read_json("{}.json".format(fname))

def _load_from_feather(fname):
    return pd.read_feather("{}.feather".format(fname))

def _load_from_stata(fname):
    return pd.read_stata("{}.dta".format(fname))

def _load_from_msgpack(fname):
    return pd.read_msgpack("{}.msgpack".format(fname))

def load(df, ctype, fname):
    s0 = time.time()
    if ctype[:7] == "parquet":
        nctype, engine, compression = ctype.split("_")
        if compression == "None":
            compression = None
        globals()["_load_from_{}".format(nctype)](fname, engine, compression)
    else:
        globals()["_load_from_{}".format(ctype)](fname)
    s1 = time.time()
    logging.info("load,{},{},{},{},{}".format(fname, ctype, s1-s0, s1, s0))

def _append_to_hdf(df, fname, key="df"):
    df.to_hdf("{}.h5".format(fname), key=key, mode="a")

def _append_to_csv(df2, fname):
    df1 = _load_from_csv(fname)
    df = pd.concat([df1, df2], sort=False)

def append(df2, ctype, fname):
    s0 = time.time()
    if ctype[:7] == "parquet":
        nctype, engine, compression = ctype.split("_")
        if compression == "None":
            compression = None
        df1 = globals()["_load_from_{}".format(nctype)](fname, engine, compression)
        df = pd.concat([df1, df2], sort=False)
        df = globals()["_append_to_{}".format(nctype)](df, fname, engine, compression)
    else:
        df1 = globals()["_load_from_{}".format(nctype)](fname, engine, compression)
        df = pd.concat([df1, df2], sort=False)
        df = globals()["_append_to_{}".format(ctype)](df2, fname)
    s1 = time.time()
    logging.info("append,{},{},{},{},{}".format(fname, ctype, s1-s0, s1, s0))

def fsize(fname):
    mypath = "."
    onlyfiles = [f for f in os.listdir(mypath) if f.startswith(str(fname)) and os.path.isfile(os.path.join(mypath, f))]
    for fn in onlyfiles:
        fsize = os.path.getsize(fn)
        logging.info("fsize,{},{:.2f},{}".format(fn, (fsize/fname*1.0), fsize ))

if __name__ == "__main__":
    ctypes = ["hdf", "csv", "pickle", "sql", "json", "feather", "stata", "msgpack"]
    parquets = []
    for engine in ['pyarrow' , 'fastparquet']:
        for compression in ['snappy', 'None']: # 'gzip', 'brotli', 'None']:
            parquets.append("parquet_{}_{}".format(engine, compression))
    ctypes += parquets
    for size in range(1,2):
        df1 = create_df_by_size(size)
        df2 = create_df_by_size(size)
        for ctype in ctypes:
            # Exception
            # fastparquet,snappy
            # Assertion failed: (PassInf && "Expected all immutable passes to be initialized"), function addImmutablePass, file /Users/buildbot/miniconda3/conda-bld/llvmdev_1545076115094/work/lib/IR/LegacyPassManager.cpp, line 812.
            # Abort trap: 6
            fname = convert(df1, ctype)
            load(df1, ctype, fname)
            #append(df2, ctype, fname)
            #fsize(fname)
