import re
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class SimilarityJoin:
    # constructor of the python class
    def __init__(self, data_file1, data_file2):
        self.df1 = pd.read_csv(data_file1)
        self.df2 = pd.read_csv(data_file2)
          
    def preprocess_df(self, df, cols): 
        # create a new column in the df to store the concatenated columns(concatenate using join)
        df['joinKey'] = df[cols].astype(str).apply(lambda x: ' '.join(x), axis=1)
        # lower the strings and replace the nan term and tokenize the string
        df['joinKey'] = df['joinKey'].str.lower().str.replace(' nan', '').str.split(r'\W+')
        # get rid of the none space, python 3 returns an iterator for filter so we need to put it into a list
        df['joinKey'] = df['joinKey'].apply(lambda x: list(filter(None, x)))
        #  return the concatenated dataframe
        return df
               
    def filtering(self, df1, df2):
        # create a column called merge and use that column to merge
        df1['merge'] = df1['joinKey']
        df2['merge'] = df2['joinKey']
        # flattern the both columns and join on that column
        df1_flattern = df1.explode('merge')
        df2_flattern = df2.explode('merge')
        # pd.merge is a database join so its runtime is not n^2
        # the suffixes will set the same column names to col_1 and col_2
        merge_df = df1_flattern.merge(df2_flattern, on='merge', suffixes=('1','2'))
        # drop the duplicates of the table based on id1 and id2
        result_df = merge_df.drop_duplicates(subset=['id1','id2'], keep='first')
        # select the designated columns
        result_df = result_df[['id1', 'joinKey1', 'id2', 'joinKey2']]
        return result_df

    def verification(self, cand_df, threshold):
        # initialize a new list to store similarity score
        result = []
        # loop for two elements at the same time
        for element1, element2, in zip(cand_df['joinKey1'], cand_df['joinKey2']):
            # convert to set and use the intersection function
            intersection = len(set(element1).intersection(set(element2)))
            # convert to set and use the union function
            union = len(set(element1).union(set(element2)))
            # append the similarity score to the result list
            result.append(float(intersection/union))
        # create a new column in cand_df and add the result list to that column
        cand_df['jaccard']=result
        # filter the cand_df on similarity score
        result_df = cand_df[cand_df['jaccard'] >= threshold]
        return result_df

    def evaluate(self, result, ground_truth):
        # flatten the 2d list to 1d so that it can be convert to set
        result_set = set([i[0]+i[1] for i in result])
        ground_truth_set = set([i[0]+i[1] for i in ground_truth])
        # calculate the true matching
        truly_matching = len(result_set.intersection(ground_truth_set))
        # calculate the precision recall and fmeasure and convert to float
        precision = float(truly_matching/len(result))
        recall = float(truly_matching/len(ground_truth))
        fmeasure =float((2*precision*recall)/(precision+recall))
        return (precision, recall, fmeasure)

    def jaccard_join(self, cols1, cols2, threshold):
        new_df1 = self.preprocess_df(self.df1, cols1)
        new_df2 = self.preprocess_df(self.df2, cols2)
        print ("Before filtering: %d pairs in total" %(self.df1.shape[0] *self.df2.shape[0])) 
        
        cand_df = self.filtering(new_df1, new_df2)
        print ("After Filtering: %d pairs left" %(cand_df.shape[0]))
        
        result_df = self.verification(cand_df, threshold)
        print ("After Verification: %d similar pairs" %(result_df.shape[0]))
        
        return result_df
       
        

if __name__ == "__main__":
    er = SimilarityJoin("Amazon_sample.csv", "Google_sample.csv")
    amazon_cols = ["title", "manufacturer"]
    google_cols = ["name", "manufacturer"]
    result_df = er.jaccard_join(amazon_cols, google_cols, 0.5)

    result = result_df[['id1', 'id2']].values.tolist()
    ground_truth = pd.read_csv("Amazon_Google_perfectMapping_sample.csv").values.tolist()
    print ("(precision, recall, fmeasure) = ", er.evaluate(result, ground_truth))