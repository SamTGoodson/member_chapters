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
       'Public Administration', 'h21p','nha21p', 'nhb21p', 'nhw21p', 'cvap21bapp']


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
    ag = df.groupby(['Precinct',district+'_choice_1'])[district+'_choice_2'].value_counts().reset_index(name='count')
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

def make_demo_map(dis_df,demo_df):
    dis_df = make_edname(dis_df)
    dis_df['ed_name'] = dis_df['ed_name'].astype('Int64')
    demo_df['ElectDist'] = demo_df['ElectDist'].astype('Int64')
    demo_map = dis_df.merge(demo_df, left_on='ed_name',right_on='ElectDist')
    demo_map['geometry'] = demo_map['geometry'].apply(wkt.loads)
    demo_map = gpd.GeoDataFrame(demo_map)
    return demo_map

def make_clustering_df(df,ed_df,cols_to_keep):
    df = make_edname(df)
    df = df.merge(ed_df, left_on='ed_name', right_on='ElectDist')
    df = df[cols_to_keep]

    return df

def scale_and_cluster(df,ed_df,cols_to_keep,n_clusters):
    df = make_clustering_df(df,ed_df,cols_to_keep)
    df = df.dropna()
    features = df.drop(columns=['ElectDist'])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(scaled_features)
    df['cluster'] = kmeans.labels_
    return df