import time
import pymysql # for pulling UCSC data
import pandas as pd
from pathlib import Path
import logging
# app
from .progress_bar import * # tqdm, context-friendly

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger('numexpr').setLevel(logging.WARNING)

# these login stats for the public database should not change.
HOST = 'genome-mysql.soe.ucsc.edu'
USER = 'genome'
DB = 'hg38'
# cpg related table schema: http://genome.ucsc.edu/cgi-bin/hgTables?db=hg38&hgta_group=regulation&hgta_track=cpgIslandExt&hgta_table=cpgIslandExt&hgta_doSchema=describe+table+schema
possible_tables = [
    'refGene', # cruzdb used this in examples -- 88,819 genes
    'knownGene', # 232,184 -- genes and pseudo genes too (use TranscriptType == 'coding_protein')
    'ncbiRefSeq', # 173,733 genes -- won't have matching descriptions; no kgXref shared key.
    # 'wgEncodeGencodeBasicV38', # 177k genes -- doesn't work
    ]

table_mapper = {
    'txStart': 'chromStart', # knownGene transcription start, refGene start, ncbiRefSeq start
    'txEnd': 'chromStart',
}
conn = None
def fetch_genes(dmr_regions_file, tol=100, ref=None, sql=None,
     use_cached=True, save=True, verbose=False,
     host=HOST, user=USER, password='', db=DB):
    f"""find genes that are adjacent to significantly different CpG regions provided.

summary:
--------
fetch_genes() annotates the DMR region output file, using the UCSC Genome Browser database as a reference
as to what genes are nearby. This is an exploratory tool, as there are many versions of the human genome
that map genes to slightly different locations.

fetch_genes() is an EXPLORATORY tool and makes a number of simplicifications:
  - the DMR regions file saves one CpG probe name and location, even though clusters of probes may map to
  that nearby area.
  - it measures the distance from the start position of the one representative probe per region to any nearby
  genes, using the `tol`erance parameter as the cutoff. Tolerance is the max number of base pairs of separation
  between the probe sequence start and the gene sequence start for it to be considered as a match.
  - The default `tol`erance is 100, but that is arbitrary. Increase it to expand the search area, or decrease it
  to be more conservative. Remember that Illumina CpG probe sequences are 50 base pairs long, so 100 is nearly
  overlapping. 300 or 500 would also be reasonable.
  - "Adjacent" in the linear sequence may not necessarily mean that the CpG island is FUNCTIONALLY coupled to the
  regulatory or coding region of the nearby protein. DNA superstructure can position regulatory elements near to
  a coding region that are far upstream or downstream from the mapped position, and there is no easy way to identify
  "adjacent" in this sense.
  - Changing the `tol`erance, or the reference database will result major differences in the output, and thus
  one's interpretation of the same data.

arguments:
----------
dmr_regions_file:
    pass in the output file from DMR function.
ref: default is `refGene`
    use one of possible_tables for lookup:
    `{possible_tables}`
tol:
    +/- this many base pairs consistutes a gene "related" to a CpG region provided.
use_cached:
    If True, the first time it downloads a dataset from UCSC Genome Browser, it will save to disk
    and use that local copy thereafter. To force it to use the online copy, set to False.
sql:
    a DEBUG mode that bypasses the function and directly queries the database for any information the user wants.
    Be sure to specify the complete SQL statement, including the ref-table (e.g. refGene or ncbiRefSeq).
host, user, password, db:
    Internal database connections for UCSC server. You would only need to mess with these of the server domain changes
    from the current hardcoded value {HOST}

 .. note::
    This method flushes cache periodically. After 30 days, it deletes cached reference gene tables and re-downloads.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
    if not sql:
        regions = pd.read_csv(dmr_regions_file) #.sort_values('z_p')
        reqd_regions = set(['name', 'chromStart'])
        if set(regions.columns) & reqd_regions != reqd_regions:
            raise KeyError(f"Your file of CpG regions must have these columns, at a minimum: {reqd_regions}")
        LOGGER.info(f"Loaded {regions.shape[0]} CpG regions from {dmr_regions_file}.")
    if not ref:
        ref = possible_tables[0] # refGene
    global conn
    if conn is None:
        conn = pymysql.connect(host=host, user=user, password=password, db=db, cursorclass=pymysql.cursors.DictCursor)
    with conn.cursor() as cur:
        if sql:
            cur.execute(sql)
            return list(cur.fetchall())
        # these will be packed into the output CSV saved, but a nested dataframe is returned.
        matches = {i:[] for i in regions.name} # cpg name --> [gene names]
        distances = {i:[] for i in regions.name}
        descriptions = {i:[] for i in regions.name}
        # fetch WHOLE table needed, unless using cache
        package_path = Path(__file__).parent
        cache_file = Path(package_path, 'data', f"{ref}.pkl")
        cache_available = cache_file.exists()
        # don't use cache if over 1 month old:
        if use_cached and cache_available:
            last_download = cache_file.stat().st_ctime
            if time.time() - last_download > 2629746:
                LOGGER.info(f"Cached genome table is over 1 month old; re-downloading from UCSC.")
                cache_file.unlink()
                cache_available = False
        if use_cached and cache_available:
            genes = pd.read_pickle(cache_file)
            LOGGER.info(f"""Using cached `{ref}`: {Path(package_path, 'data', f"{ref}.pkl")} with ({len(genes)}) genes""")
        else: # download it
            LOGGER.info(f"Downloading {ref}")
            # chrom, txStart, txEnd; all 3 tables have name, but knownGene lacks a name2.
            if ref == 'knownGene':
                sql = f"""SELECT name as name2, txStart, txEnd, description FROM {ref} LEFT JOIN kgXref ON kgXref.kgID = {ref}.name;"""
            else:
                sql = f"""SELECT name, name2, txStart, txEnd, description FROM {ref} LEFT JOIN kgXref ON kgXref.refseq = {ref}.name;"""
            cur.execute(sql)
            genes = list(cur.fetchall())
            if use_cached:
                import pickle
                with open(Path(package_path, 'data', f"{ref}.pkl"),'wb') as f:
                    pickle.dump(genes, f)
                    LOGGER.info(f"Cached {Path(package_path, 'data', f'{ref}.pkl')} on first use, with {len(genes)} genes")
            else:
                LOGGER.info(f"Using {ref} with {len(genes)} genes")
        # compare two dataframes and calc diff.
        # need to loop here: but prob some matrix way of doing this faster
        done = 0
        for gene in tqdm(genes, total=len(genes), desc="Mapping genes"):
            closeby = regions[ abs(regions.chromStart - gene['txStart']) < tol ]
            if len(closeby) > 0:
                for idx,item in closeby.iterrows():
                    matches[item['name']].append(gene['name2'])
                    dist = item['chromStart'] - gene['txStart']
                    distances[item['name']].append(dist)
                    desc = gene['description'].decode('utf8') if gene['description'] != None else ''
                    descriptions[item['name']].append(desc)
                    done += 1
                    #if done % 1000 == 0:
                    #    LOGGER.info(f"[{done} matches]")
    # also, remove duplicate gene matches for the same region (it happens a lot)
    matches = {k: ','.join(set(v)) for k,v in matches.items()}
    distances = {k: ','.join(set([str(j) for j in v])) for k,v in distances.items()}
    descriptions = {k: ' | '.join(set(v)) for k,v in descriptions.items()}
    # tidying up some of the deduping
    def _tidy(desc):
        if desc.startswith('|'):
            desc = desc.lstrip('|')
        if desc.endswith('|'):
            desc = desc.rstrip('|')
        return desc
    descriptions = {k: _tidy(desc) for k,desc in descriptions.items()}
    regions['genes'] = regions['name'].map(matches)
    regions['distances'] = regions['name'].map(distances)
    regions['descriptions'] = regions['name'].map(descriptions)
    #finaly, add column to file and save
    if save:
        dmr_regions_stem = str(dmr_regions_file).replace('.csv','')
        outfile = f"{dmr_regions_stem}_genes.csv"
        regions.to_csv(Path(outfile))
        LOGGER.info(f"Wrote {outfile}")
    return regions
