import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


class AnomalyDetection():
    
    def get_elements(self, df_row, indices):
        elments = []
        for i in indices:
            elments.append(df_row[i])
        return elments 

    # encode the categorical features to numerical features
    def encode(self, df_row, positions, indices):
        encode = [0]*len(positions)
        for i in indices:
            encode[positions.index(df_row[i])] = 1
        # append the remaing numerical features into the new feature vector
        max_pos = max(indices)
        encode.extend((df_row[max_pos+1:]))
        return encode
    
    def cat2Num(self, df, indices):
        # get all the categorical features
        elements = df['features'].apply(self.get_elements,args=(indices,))
        # flatten the categorical features into on list
        flatten = []
        for i in elements:
            for j in i:
                flatten.append(j)
        # take the unique categoricals
        # As I read  the categoricals row by row so the encoding results I got is different from the given example
        # But the result I got followed the encoding rules, I have double checked with TA Andy and he said this is ok.
        positions = np.unique(flatten).tolist()
        df['encode'] = df['features'].apply(self.encode, args=(positions,indices,))
        df = df [['encode']]
        df = df.rename(columns={'encode':'features'})        
        return df

    def scaleNum(self, df, indices):
        for i in indices:
            # store all the elements needed in one list 
            features = [element[i] for element in df['features']]
            # convert to a array so that we can use np.mean and np.std
            features = np.asarray(features)
            # set ddof=1 so that we get the unbiased std
            std = np.std(features,ddof=1)
            mean = np.mean(features)
            # standarization
            for element in df['features']:
                element[i] = (element[i]-mean)/std
        return df

    def detect(self, df, k, t):
        # use Kmeans to cluster the data set 
        df['cluster'] = KMeans(n_clusters=k, random_state=0).fit_predict(df['features'].to_list())
        # group by the dataset so that we got n_max and n_min
        clusters_count = df.groupby('cluster').agg('count')
        
        clusters_count = clusters_count.rename(columns={'features':'count'})
        n_max = clusters_count['count'].max()
        n_min = clusters_count['count'].min()
        # merge the groupby dataset back to the original dataset so that we got the Nx for each record
        df = df.merge(clusters_count, on='cluster',suffixes=['_original', '_group']).reset_index()
        df['score'] = (n_max-df['count'])/(n_max-n_min)
        # filter the dataset with threshold
        df = df[df['score']>0.9]
        # rename the columns
        df = df[['index','features','score']]
        df = df.rename(columns={'index':'id'})
        df = df.set_index('id')
        return df


if __name__ == "__main__":
    # convert the string to a list object so that we can use index to access
    df = pd.read_csv('logs-features-sample.csv', converters={'features': eval}).set_index('id')

    ad = AnomalyDetection()
    
    df1 = ad.cat2Num(df, [0,1])
    print(df1)

    df2 = ad.scaleNum(df1, [6])
    print(df2)

    df3 = ad.detect(df2, 8, 0.97)
    print(df3)