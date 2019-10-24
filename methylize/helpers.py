
def map_to_genome(df, rgset):
    """Maps dataframe to genome locations
    Parameters
    ----------
    df: dataframe
            Dataframe containing methylation, unmethylation, M or Beta
            values for each sample at each site
    rgset: rg channel set instance
            RG channel set instance related to provided df
    Returns
    -------
    df: dataframe
            Dataframe containing the original values with the addition
            of genomic locations for each site
    """
    # This (mapToGenome) was in an old copy of methpype, but moved to methylize where it might be used.
    if 'Name' in df.columns:
        mani = rgset.manifest
        chromosomes = dict(
            zip(list(mani['Name'].values), list(mani['CHR'].values)))
        strands = dict(
            zip(list(mani['Name'].values), list(mani['Strand'].values)))
        build = dict(zip(list(mani['Name'].values),
                         list(mani['Genome_Build'].values)))
        mapinfo = dict(
            zip(list(mani['Name'].values), list(mani['MAPINFO'].values)))
        df['Chr'] = df['Name'].map(chromosomes)
        df['Strand'] = df['Name'].map(strands)
        df['GenomeBuild'] = df['Name'].map(build)
        df['MapInfo'] = df['Name'].map(mapinfo)
        return df
    else:
        print("No 'Name' column in dataframe.")
        return

def readManifest(array):
    """DEPRECRATED VERSION -- Returns Illumina manifest for array type
    Parameters
    ----------
    array: str
        String specifying the type of Illumina Methylation Array
    Returns
    -------
    manifest: dataframe
        Dataframe containing Illumina Human Methylation Array manifest
    """
    # Copy of manifest-reading code from old version of methpype
    downloadManifest(array)
    if array == 'CustomArray':
        manifest = pd.read_csv(os.path.expanduser(
            "~/.methpype_manifest_files/CombinedManifestEPIC.manifest.CoreColumns.csv.gz"))
        # TEMPORARY - Remove missing probes
        manifest = manifest[manifest['AddressA_ID'] != 9614306]
        manifest = manifest[manifest['AddressA_ID'] != 9637526]
        manifest = manifest[manifest['AddressA_ID'] != 3680876]
        manifest = manifest[manifest['AddressB_ID'] != 60646183]
    elif array == 'IlluminaHumanMethylation450k':
        #manifest = pd.read_csv(os.path.expanduser("~/.methpype_manifest_files/HumanMethylation450_15017482_v1-2.CoreColumns.csv.gz"))
        manifest = pd.read_csv("~/HumanMethylation450_15017482_v1-2.csv")
    elif array == 'IlluminaHumanMethylationEPIC':
        manifest = pd.read_csv(os.path.expanduser(
            "~/.methpype_manifest_files/MethylationEPIC_v-1-0_B4.CoreColumns.csv.gz"))
    manifest = manifest.rename({'AddressA_ID': 'AddressA', 'AddressB_ID': 'AddressB', 'Infinium_Design_Type': 'Type',
                                'Color_Channel': 'Color', 'IlmnID': 'CtrlAddress', 'Name': 'Name'}, axis='columns')
    manifest.dropna(how='all', inplace=True)
    manifest = flagProbeTypes(manifest)
    # CURRENTLY UNUSED CODE
    # KEEP FOR FUTURE ITERATION WHERE PROBE SEQUENCES CAN BE MAPPED BACK
    # # determine nCpGs
    # nCpGs = {}
    # for name,A,B,T in zip(manifest['Name'],manifest['ProbeSeqA'],manifest['ProbeSeqB'],manifest['Type']):
    #   if T == 'I':
    #       nCpGs[name] = int(B.count("CG")-1)
    #   elif T == 'II':
    #       nCpGs[name] = int(A.count('R'))
    #   else:
    #       nCpGs[name] = int(0)
    # # add to dataframe
    # manifest['nCpG'] = manifest['Name'].map(nCpGs)
    # determine types
    return manifest
