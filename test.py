
from fastapi import APIRouter,HTTPException,File, Query
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import pickle
from sklearn.metrics import pairwise_distances_argmin_min
import random




router = APIRouter()
@router.get("/ML_Project/")

async def script(user_name: str, playlist_link: str):

    playlist_features_list = ["artist","album","track_name",  "track_id","danceability","energy","key","loudness","mode", "speechiness","instrumentalness","liveness","valence","tempo", "duration_ms","time_signature"]
    
    playlist_df = pd.DataFrame(columns = playlist_features_list)

    cid = '91382d101a194c12b67ba7f857bede3e'
    secret = '74179f9842234459a51e85c7f65bd180'

    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)

    sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

    playlist = 'https://open.spotify.com/playlist/7p8KeDuRmErO5J8f8BUyk9?si=b7377b96d49c4e8c'
    username = 'tss'
    #step2
    owner = sp.user_playlist(user_name,playlist_link)['owner']['display_name']
    playlist_name = sp.user_playlist(user_name,playlist_link)['name']
    playlist = sp.user_playlist_tracks(user_name, playlist_link)["items"]
    for track in playlist:
        # Create empty dict
        playlist_features = {}
        # Get metadata
        try:
            playlist_features["artist"] = track["track"]["album"]["artists"][0]["name"]
            playlist_features["album"] = track["track"]["album"]["name"]
            playlist_features["track_name"] = track["track"]["name"]
            playlist_features["track_id"] = track["track"]["id"]

            # Get audio features
            audio_features = sp.audio_features(playlist_features["track_id"])[0]
            for feature in playlist_features_list[4:]:
                playlist_features[feature] = audio_features[feature]

            # Concat the dfs
            track_df = pd.DataFrame(playlist_features, index = [0])
            playlist_df = pd.concat([playlist_df, track_df], ignore_index = True)
            playlist_df["owner"] = owner
            playlist_df["playlist_name"] = playlist_name
            playlist_df["user"] = user_name
        
        except:
            pass
    #print(playlist_df)
    df = playlist_df.copy()

    col = ['playlist_name', 'user','danceability', 'energy',
       'key', 'loudness', 'mode', 'speechiness', 'instrumentalness',
       'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']

    columns = ['playlist_name', 'user']
    for n in col[2:]:
        columns.append(n+'_var')
        columns.append(n+'_mean')
        columns.append(n+'_max')
        columns.append(n+'_min')
    

    data = pd.DataFrame(columns = columns)



    dict = {'playlist_name':df['playlist_name'].iloc[0], 'user':df['user'].iloc[0]}
    for i in col[2:]:
        Var = i+'_var'
        dict[Var] = df[i].var()
        Mean = i+'_mean'
        dict[Mean] = df[i].mean()
        Max = i+'_max'
        dict[Max] = max(df[i])
        Min = i+'_min'
        dict[Min] = min(df[i])
    data = data.append(dict, ignore_index=True)

    #import newdata from github
    dataset = pd.read_csv("newdata.csv")
    dataset = dataset.append(data, ignore_index=True)
    dataset2 = dataset.drop(columns=['playlist_name', 'user'])
    norm_data = normalize(dataset2)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(norm_data)
    pc_data = pd.DataFrame(data = principalComponents)
    new_sample = pc_data.loc[252,:]
    pc_data = pc_data.drop([252])
    new_sample = pd.DataFrame(new_sample).T

    #import model from github
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
        cluster_array = model.predict(new_sample)
        cluster = cluster_array[0]
        pc_data['cluster'] = model.labels_
        closest, _ = pairwise_distances_argmin_min(new_sample.to_numpy().reshape(1,-1),pc_data[pc_data['cluster'] == cluster].iloc[:,0:2])
        friend = dataset.loc[closest[0],:]['user']
        f_pl = dataset.loc[closest[0],:]['playlist_name']

    print(friend, f_pl)


    return {'Friend':friend,'Friend Playlist':f_pl}