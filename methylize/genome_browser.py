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
def fetch_genes(dmr_regions_file=None, tol=250, ref=None, tissue=None, sql=None,
    save=True, verbose=False, use_cached=True, no_sync=False, genome_build=None,
    host=HOST, user=USER, password='', db=DB):
    """find genes that are adjacent to significantly different CpG regions provided.

Summary:

    fetch_genes() annotates the DMR region output file, using the UCSC Genome Browser database as a reference
    as to what genes are nearby. This is an exploratory tool, as there are many versions of the human genome
    that map genes to slightly different locations.

fetch_genes() is an EXPLORATORY tool and makes a number of simplicifications:

  * the DMR regions file saves one CpG probe name and location, even though clusters of probes may map to
    that nearby area.
  * it measures the distance from the start position of the one representative probe per region to any nearby
    genes, using the `tol`erance parameter as the cutoff. Tolerance is the max number of base pairs of separation
    between the probe sequence start and the gene sequence start for it to be considered as a match.
  * The default `tol`erance is 250, but that is arbitrary. Increase it to expand the search area, or decrease it
    to be more conservative. Remember that Illumina CpG probe sequences are 50 base pairs long, so 100 is nearly
    overlapping. 300 or 500 would also be reasonable.
  * "Adjacent" in the linear sequence may not necessarily mean that the CpG island is FUNCTIONALLY coupled to the
    regulatory or coding region of the nearby protein. DNA superstructure can position regulatory elements near to
    a coding region that are far upstream or downstream from the mapped position, and there is no easy way to identify
    "adjacent" in this sense.
  * Changing the `tol`erance, or the reference database will result major differences in the output, and thus
    one's interpretation of the same data.
  * Before interpreting these "associations" you should also consider filtering candidate genes by
    specific cell types where they are expressed. You should know the tissue from which your samples originated.
    And filter candidate genes to exclude those that are only expressed in your tissue during development,
    if your samples are from adults, and vice versa.

Arguments:

    dmr_regions_file:
        pass in the output file DataFrame or FILEPATH from DMR function.
        Omit if you specify the `sql` kwarg instead.
    ref: default is `refGene`
        use one of possible_tables for lookup:
        - 'refGene' -- 88,819 genes -- default table used in comb-b and cruzdb packages.
        - 'knownGene' -- 232,184 genes -- pseudo genes too (the "WHere TranscriptType == 'coding_protein'" clause would work, but these fields are missing from the data returned.)
        - 'ncbiRefSeq' -- 173,733 genes -- this table won't have gene descriptions, because it cannot be joined with the 'kgXref' (no shared key).
        Additionally, 'gtexGeneV8' is used for tissue-expression levels. Pseudogenes are ommited using the "WHERE score > 0" clause in the SQL.

    tol: default 250
        +/- this many base pairs consistutes a gene "related" to a CpG region provided.
    tissue: str
        if specified, adds additional columns to output with the expression levels for identified genes
        in any/all tissue(s) that match the keyword. (e.g. if your methylation samples are whole blood,
        specify `tissue=blood`) For all 54 tissues, use `tissue=all`
    genome_build: (None, NEW, OLD)
        Only the default human genome build, hg38, is currently supported. Even though many other builds are available
        in the UCSC database, most tables do not join together in the same way.
    use_cached:
        If True, the first time it downloads a dataset from UCSC Genome Browser, it will save to disk
        and use that local copy thereafter. To force it to use the online copy, set to False.
    no_sync:
        methylize ships with a copy of the relevant UCSC gene browser tables, and will auto-update these
        every month. If you want to run this function without accessing this database, you can avoid updating
        using the `no_sync=True` kwarg.
    host, user, password, db:
        Internal database connections for UCSC server. You would only need to mess with these of the server domain changes
        from the current hardcoded value {HOST}. Necessary for tables to be updated and for `tissue` annotation.
    sql:
        a DEBUG mode that bypasses the function and directly queries the database for any information the user wants.
        Be sure to specify the complete SQL statement, including the ref-table (e.g. refGene or ncbiRefSeq).

.. note::
   This method flushes cache periodically. After 30 days, it deletes cached reference gene tables and re-downloads.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
    if isinstance(dmr_regions_file, pd.DataFrame):
        regions = dmr_regions_file
        reqd_regions = set(['name', 'chromStart'])
        if set(regions.columns) & reqd_regions != reqd_regions:
            raise KeyError(f"Your file of CpG regions must have these columns, at a minimum: {reqd_regions}")
        LOGGER.info(f"Loaded {regions.shape[0]} CpG regions.")
    elif not sql and dmr_regions_file is None:
        raise Exception("Either provide a path to the DMR stats file or a sql query.")
    elif not sql:
        regions = pd.read_csv(dmr_regions_file) #.sort_values('z_p')
        reqd_regions = set(['name', 'chromStart'])
        if set(regions.columns) & reqd_regions != reqd_regions:
            raise KeyError(f"Your file of CpG regions must have these columns, at a minimum: {reqd_regions}")
        LOGGER.info(f"Loaded {regions.shape[0]} CpG regions from {dmr_regions_file}.")
    if not ref:
        ref = possible_tables[0] # refGene

    global conn # allows function to reuse the same connection
    if conn is None and no_sync is False:
        conn = pymysql.connect(host=host, user=user, password=password, db=db, cursorclass=pymysql.cursors.DictCursor)

    if sql:
        with conn.cursor() as cur:
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
    if use_cached and cache_available and no_sync is False:
        last_download = cache_file.stat().st_ctime
        if time.time() - last_download > 2629746:
            LOGGER.info(f"Cached genome table is over 1 month old; re-downloading from UCSC.")
            cache_file.unlink()
            cache_available = False
    if use_cached and cache_available:
        genes = pd.read_pickle(cache_file)
        LOGGER.info(f"""Using cached `{ref}`: {Path(package_path, 'data', f"{ref}.pkl")} with ({len(genes)}) genes""")
    elif no_sync is False: # download it
        LOGGER.info(f"Downloading {ref}")
        # chrom, txStart, txEnd; all 3 tables have name, but knownGene lacks a name2.
        if ref == 'knownGene':
            sql = f"""SELECT name as name2, txStart, txEnd, description FROM {ref} LEFT JOIN kgXref ON kgXref.kgID = {ref}.name;"""
        else:
            sql = f"""SELECT name, name2, txStart, txEnd, description FROM {ref} LEFT JOIN kgXref ON kgXref.refseq = {ref}.name;"""
        with conn.cursor() as cur:
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

    # add column(s) for gene tissue expression
    if tissue != None:
        # tissue == 'all'
        tissues = fetch_genes(sql="select * from hgFixed.gtexTissueV8;")
        sorted_tissues = [i['name'] for i in tissues]
        gene_names = [i.split(',') for i in list(regions['genes']) if i != '']
        N_regions_with_multiple_genes = len([i for i in gene_names if len(i) > 1])
        if N_regions_with_multiple_genes > 0:
            LOGGER.warning(f"{N_regions_with_multiple_genes} of the {len(gene_names)} regions have multiple genes matching in the same region, and output won't show tissue expression levels.")
        gene_names = tuple([item for sublist in gene_names for item in sublist])
        gtex = fetch_genes(sql=f"select name, expScores from gtexGeneV8 WHERE name in {gene_names} and score > 0;")
        if len(gtex) > 0:
            # convert to a lookup dict of gene name: list of tissue scores
            gtex = {item['name']: [float(i) for i in item['expScores'].decode().split(',') if i != ''] for item in gtex}

            # add tissue names
            if len(tissues) != len(list(gtex.values())[0]):
                LOGGER.error(f"GTEx tissue names and expression levels mismatch.")
            else:
                for gene, expScores in gtex.items():
                    labeled_scores = dict(zip(sorted_tissues, expScores))
                    gtex[gene] = labeled_scores
                # to merge, create a new dataframe with matching genes names as index.
                tissue_df = pd.DataFrame.from_dict(data=gtex, orient='index')
                if tissue != 'all':
                    matchable = dict(zip([k.lower() for k in list(tissue_df.columns)], list(tissue_df.columns)))
                    keep_columns = [col_name for item,col_name in matchable.items() if tissue.lower() in item]
                    if keep_columns == []:
                        LOGGER.warning(f"No GTEx tissue types matched: {tissue}; returning all tissues instead.")
                    else:
                        tissue_df = tissue_df[keep_columns]
                # this merge will ONLY WORK if there is just one gene listed in the gene column
                regions = regions.merge(tissue_df, how='left', left_on='genes', right_index=True)

    #finaly, add column to file and save
    if save:
        dmr_regions_stem = str(dmr_regions_file).replace('.csv','')
        outfile = f"{dmr_regions_stem}_genes.csv"
        regions.to_csv(Path(outfile))
        LOGGER.info(f"Wrote {outfile}")
    return regions

"""
tissue='all' (for big table) or tissue='blood' for one extra column
TODO -- incorporate the GTEx tables (expression by tissue) if user specifies one of 54 tissue types covered.

gtexGeneV8 x gtexTissue

"hgFixed.gtexTissue lists each of the 53 tissues in alphabetical order, corresponding to the comma separated expression values in gtexGene."

works: tissue_lookup = m.fetch_genes('', sql="select * from hgFixed.gtexTissueV8;")
then match tissue keyword kwarg against 'description' field and use 'name' for colname

note that expScores is a list of 54 numbers (expression levels).

chrom	chromStart	chromEnd	name	score	strand	geneId	geneType	expCount	expScores
{'chrom': 'chr1',
  'chromEnd': 29806,
  'chromStart': 14969,
  'expCount': 53,
  'expScores': b'6.886,6.083,4.729,5.91,6.371,6.007,8.768,4.202,4.455,4.64,10'
               b'.097,10.619,6.108,5.037,5.018,4.808,4.543,4.495,5.576,4.57,8'
               b'.275,4.707,2.55,9.091,9.885,8.17,7.392,7.735,5.353,7.124,8.6'
               b'17,3.426,2.375,7.669,3.826,7.094,6.365,3.263,10.723,10.507,4'
               b'.843,9.193,13.25,11.635,11.771,8.641,10.448,6.522,9.313,10.3'
               b'04,9.987,9.067,6.12,',
  'geneId': 'ENSG00000227232.4',
  'geneType': 'unprocessed_pseudogene',
  'name': 'WASH7P',
  'score': 427,
  'strand': '-'},
"""
