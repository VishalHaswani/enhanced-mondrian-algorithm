import anonypy
import datetime
import pandas as pd
from generate_fake_dataset import dataset

if __name__ == '__main__':
    K = 4  # k-anonymity

    # sample dataset
    columns, feature_columns, sensitive_column, data = dataset()
    start_time = datetime.datetime.now()

    df = pd.DataFrame(data=data, columns=columns)
    df['gender'] = df['gender'].astype('category')
    p = anonypy.Preserver(df, feature_columns, sensitive_column)
    rows = p.anonymize_k_anonymity(k=K)
    anonymized_dataframe = pd.DataFrame(rows, columns=columns)

    end_time = datetime.datetime.now() 
    tdelta = end_time - start_time
    print("Total time taken:", tdelta)
    
    print('#'*43 + '\nTop 30 rows of the dataset:\n', anonymized_dataframe.head(30))
    print('#'*43 + '\nDealing with Catagorical Values\n', anonymized_dataframe[(anonymized_dataframe['gender'] != 'M') & (anonymized_dataframe['gender'] != 'F')])
    print('#'*43 + '\nFinal Dataset\n', anonymized_dataframe)