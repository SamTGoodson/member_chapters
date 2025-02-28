import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import wkt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

job_cols = ['Construction', 'Manufacturing', 'Wholesale Trade',
       'Retail Trade', 'Transportation and Warehousing', 'Information',
       'Finance and Insurance',
       'Professional, Scientific, and Technical Services',
       'Management of Companies and Enterprises', 'Educational Services',
       'Health Care and Social Assistance',
       'Arts, Entertainment, and Recreation',
       'Accommodation and Food Services']
race_cols = ['h21p', 'nha21p', 'nhb21p', 'nhw21p']
cols_to_keep = ['ElectDist','mhhi21','Construction', 'Manufacturing', 'Wholesale Trade', 'Retail Trade',
       'Transportation and Warehousing', 'Information',
       'Finance and Insurance', 'Real Estate and Rental and Leasing',
       'Professional, Scientific, and Technical Services',
       'Management of Companies and Enterprises',
       'Administrative and Support and Waste Management and Remediation Services',
       'Educational Services', 'Health Care and Social Assistance',
       'Arts, Entertainment, and Recreation',
       'Accommodation and Food Services',
       'Other Services [except Public Administration]',
       'Public Administration', 'nhb21p', 'cvap21bapp','white_transplant_ratio']

brooklyn_zips = ['11201',
 '11203',
 '11204',
 '11205',
 '11206',
 '11207',
 '11208',
 '11209',
 '11210',
 '11211',
 '11212',
 '11213',
 '11214',
 '11215',
 '11216',
 '11217',
 '11218',
 '11219',
 '11220',
 '11221',
 '11222',
 '11223',
 '11224',
 '11225',
 '11226',
 '11228',
 '11229',
 '11230',
 '11231',
 '11232',
 '11233',
 '11234',
 '11235',
 '11236',
 '11237',
 '11238',
 '11239',
 '11241',
 '11243',
 '11249']

zip_list = ['10001',
 '10119',
 '10199',
 '10002',
 '10003',
 '10026',
 '10004',
 '10005',
 '10271',
 '10006',
 '10007',
 '10278',
 '10279',
 '10009',
 '10010',
 '10011',
 '10012',
 '10013',
 '10014',
 '10016',
 '10017',
 '10165',
 '10167',
 '10168',
 '10169',
 '10170',
 '10171',
 '10172',
 '10173',
 '10174',
 '10177',
 '10018',
 '10030',
 '10019',
 '10020',
 '10103',
 '10111',
 '10112',
 '10021',
 '10022',
 '10152',
 '10153',
 '10154',
 '10023',
 '10024',
 '10025',
 '10027',
 '10115',
 '10028',
 '10029',
 '10031',
 '10032',
 '10033',
 '10034',
 '10037',
 '10035',
 '10036',
 '10110',
 '10038',
 '10040',
 '11104',
 '11105',
 '10039',
 '10128',
 '10044',
 '10065',
 '10069',
 '10075',
 '10162',
 '10280',
 '10282',
 '10301',
 '11106',
 '10302',
 '10303',
 '10304',
 '10451',
 '10305',
 '10306',
 '10307',
 '10308',
 '10309',
 '10310',
 '10312',
 '10311',
 '10314',
 '10456',
 '10452',
 '10453',
 '10454',
 '10455',
 '10457',
 '10458',
 '10459',
 '10460',
 '10461',
 '11004',
 '11005',
 '11040',
 '11101',
 '10462',
 '10463',
 '10464',
 '10465',
 '10466',
 '10467',
 '10468',
 '10469',
 '10470',
 '11102',
 '11109',
 '10471',
 '10472',
 '10473',
 '10474',
 '11103',
 '10475',
 '11201',
 '11203',
 '11204',
 '11205',
 '11206',
 '11207',
 '11209',
 '11425',
 '11208',
 '11210',
 '11211',
 '11212',
 '11213',
 '11218',
 '11214',
 '11215',
 '11216',
 '11217',
 '11219',
 '11220',
 '11221',
 '11222',
 '11223',
 '11224',
 '11225',
 '11226',
 '11228',
 '11230',
 '11415',
 '11424',
 '11229',
 '11231',
 '11358',
 '11232',
 '11233',
 '11234',
 '11235',
 '11236',
 '11237',
 '11238',
 '11239',
 '11354',
 '11355',
 '11356',
 '11351',
 '11357',
 '11359',
 '11360',
 '11372',
 '11361',
 '11362',
 '11416',
 '11363',
 '11364',
 '11373',
 '11365',
 '11366',
 '11367',
 '11374',
 '11368',
 '11417',
 '11369',
 '11371',
 '11370',
 '11375',
 '11379',
 '11377',
 '11378',
 '11421',
 '11422',
 '11385',
 '11003',
 '11411',
 '11412',
 '11413',
 '11414',
 '11420',
 '11418',
 '11419',
 '11423',
 '11426',
 '11427',
 '11428',
 '11436',
 '11691',
 '11692',
 '11693',
 '11001',
 '11429',
 '11432',
 '11433',
 '11451',
 '11430',
 '11434',
 '11435',
 '11694',
 '11697',
 '99999']

def aggregate_precinct_counts(df,district,candidate1,candidate2):
    mask_candidate1_candidate2 = (df[district+'_choice_1'] == candidate1) & (df[district+'_choice_2'] == candidate2)
    mask_candidate1_notcandidate2 = (df[district+'_choice_1'] == candidate1) & (df[district+'_choice_2'] != candidate2) 
    mask_candidate2_candidate1 = (df[district+'_choice_1'] == candidate2) & (df[district+'_choice_2'] == candidate1)
    mask_candidate2_notcandidate1 = (df[district+'_choice_1'] == candidate2) & (df[district+'_choice_2'] != candidate1)
    mask_neither = (df[district+'_choice_1'] != candidate1) & (df[district+'_choice_1'] != candidate2)

    

    df['candidate1_candidate2'] = df['count'] * mask_candidate1_candidate2.astype(int)
    df['candidate1_notcandidate2'] = df['count'] * mask_candidate1_notcandidate2.astype(int)
    df['candidate2_candidate1'] = df['count'] * mask_candidate2_candidate1.astype(int)
    df['candidate2_notcandidate1'] = df['count'] * mask_candidate2_notcandidate1.astype(int)
    df['neither'] = df['count'] * mask_neither.astype(int)

    result = df.groupby('Precinct')[['candidate1_candidate2', 'candidate1_notcandidate2', 'candidate2_candidate1', 'candidate2_notcandidate1','neither']].sum().reset_index()
    result.columns = ['Precinct',f'{candidate1}_{candidate2}', f'{candidate1}_not{candidate2}', f'{candidate2}_{candidate1}', f'{candidate2}_not{candidate1}','neither']
    
    return result

def make_edname(df):
    df['AD'] = df['Precinct'].str.split(' ').str[1]
    df['ED'] = df['Precinct'].str.split(' ').str[3]
    df['ed_name'] = df['AD'].astype(str) + df['ED'].astype(str).str.zfill(3)
    df['ed_name'] = df['ed_name'].astype('Int64')
    return df

def find_biggest_col(row):
    return row.idxmax()

def make_precinct_counts(df,district,candidate1,candidate2):
    ag = df.groupby(['Precinct',district+'_choice_1'])[district+'_choice_2'].value_counts(dropna=False).reset_index(name='count')
    pc = aggregate_precinct_counts(ag,district,candidate1,candidate2)
    cols = pc.columns.tolist()
    cols.remove('Precinct')
    cols.remove('neither')
    pc['biggest'] = pc[cols].apply(find_biggest_col,axis=1)
    pc = make_edname(pc)
    return pc

def make_pc_map(df,gdf,district,candidate1,candidate2):
        precinct_counts = make_precinct_counts(df,district,candidate1,candidate2)
        dis_map = precinct_counts.merge(gdf, left_on='ed_name', right_on='ElectDist')
        dis_map['biggest'] = dis_map['biggest'].astype('category')
        dis_map = gpd.GeoDataFrame(dis_map)
        return dis_map

def map_round_votes(df,gdf,district,candidate1,candidate2):
    dis_map = make_pc_map(df,gdf,district,candidate1,candidate2)
    dis_map.plot(column='biggest', cmap='tab10', legend=True, figsize=(15, 10))
    plt.title(f"Early Round Voting Direction in {district}")
    plt.show()

def make_demo_map(eds_list,demo_df):
    demo_map = demo_df[demo_df['ElectDist'].isin(eds_list)]
    demo_map['geometry'] = demo_map['geometry'].apply(wkt.loads)
    demo_map = gpd.GeoDataFrame(demo_map)
    return demo_map

def make_clustering_df(df,ed_df,cols_to_keep):
    df = make_edname(df)
    df = df.merge(ed_df, left_on='ed_name', right_on='ElectDist')
    df = df[cols_to_keep]

    return df

def scale_and_cluster(df,n_clusters):
    df = df.dropna()
    features = df.drop(columns=['ElectDist'])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(scaled_features)
    df['cluster'] = kmeans.labels_
    return df