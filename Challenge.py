# import dependencies
import json
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from sqlalchemy import update
import psycopg2
from config import db_password
import time

def ETL(wiki, kaggle, rating):
    file_dir = 'C:/Users/knush/GitRepository/Movies_ETL'
    # with open(f'{file_dir}/wikipedia.movies.json', mode='r') as file:
    with open(f'{file_dir}/{wiki}', mode='r') as file:
    # with open(wiki, mode = 'r') as file:
        wiki_movies_raw = json.load(file)
        #print(wiki_movies_raw)
        #print("********")
    #kaggle_metadata = pd.read_csv(f'{file_dir}/movies_metadata.csv', low_memory=False)
    kaggle_metadata = pd.read_csv(f'{file_dir}/{kaggle}', low_memory=False)
    #print(kaggle_metadata)
    #movie_ratings = pd.read_csv(f'{file_dir}/ratings.csv')
    ratings = pd.read_csv(f'{file_dir}/{rating}')
    #print(ratings)

    wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie
                   and 'No. of episodes' not in movie]

    def clean_movie(movie):
        movie = dict(movie) #create a non-destructive copy
        alt_titles = {} # Add empty dictionary to hold alternate titles
        # combine alternate titles into one list
        for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                    'Hangul','Hebrew','Hepburn','Japanese','Literally',
                    'Mandarin','McCune–Reischauer','Original title','Polish',
                    'Revised Romanization','Romanized','Russian',
                    'Simplified','Traditional','Yiddish']:
            if key in movie: # Check if the current key exists in the movie object.
                alt_titles[key] = movie[key]
                movie.pop(key) #if the current key exists, remove the key-value pair and add to the alternative titles dictionary.
        if len(alt_titles) > 0:
            movie['alt_titles'] = alt_titles # After looping through every key, add the alternative titles dict to the movie object.


        # merge column names
        def change_column_name(old_name, new_name):
            if old_name in movie:
                movie[new_name] = movie.pop(old_name)
        change_column_name('Adaptation by', 'Writer(s)')
        change_column_name('Country of origin', 'Country')
        change_column_name('Directed by', 'Director')
        change_column_name('Distributed by', 'Distributor')
        change_column_name('Edited by', 'Editor(s)')
        change_column_name('Length', 'Running time')
        change_column_name('Original release', 'Release date')
        change_column_name('Music by', 'Composer(s)')
        change_column_name('Produced by', 'Producer(s)')
        change_column_name('Producer', 'Producer(s)')
        change_column_name('Productioncompanies ', 'Production company(s)')
        change_column_name('Productioncompany ', 'Production company(s)')
        change_column_name('Released', 'Release Date')
        change_column_name('Release Date', 'Release date')
        change_column_name('Screen story by', 'Writer(s)')
        change_column_name('Screenplay by', 'Writer(s)')
        change_column_name('Story by', 'Writer(s)')
        change_column_name('Theme music composer', 'Composer(s)')
        change_column_name('Written by', 'Writer(s)')
        return movie
    # Create clean_movies list comprehension
    clean_movies = [clean_movie(movie) for movie in wiki_movies]
    # Set wiki_movies_df to the DataFrame created from clean_movies
    wiki_movies_df = pd.DataFrame(clean_movies)
    
    try:
        # Remove the duplicate rows
        wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
        #print(len(wiki_movies_df))
        wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
        #print(len(wiki_movies_df))
    except:
        print("Error in imbd_link")

    # Remove columns that are 90% null
    wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
    wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]

    # drops missing values from box_office column 
    box_office = wiki_movies_df['Box office'].dropna()
    # To use regular expressions convert the box office data to string - using lambda method
    lambda x: type(x) != str
    box_office[box_office.map(lambda x: type(x) != str)]
    # Applying lambda method to box_office data
    box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)

    # remove a space \*s & misspelled "millon"
    form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
    # Fix Pattern matches
    form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'
    # Some values are given as a range. To solve this problem, use search for any string that starts with a dollar sign 
    # and ends with a hyphen, and then replace it with just a dollar sign using the replace() method. 
    box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
    # Extract and convert box office values
    box_office.str.extract(f'({form_one}|{form_two})')

    # Turn the extracted values to numeric values
    def parse_dollars(s):
        # if s is not a string, return NaN
        if type(s) != str:
            return np.nan
        # if input is of the form $###.# million
        if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):
            # remove dollar sign and " million"
            s = re.sub('\$|\s|[a-zA-Z]','', s)
            # convert to float and multiply by a million
            value = float(s) * 10**6
            # return value
            return value
        # if input is of the form $###.# billion
        elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):
            # remove dollar sign and " billion"
            s = re.sub('\$|\s|[a-zA-Z]','', s)
            # convert to float and multiply by a billion
            value = float(s) * 10**9
            # return value
            return value
        # if input is of the form $###,###,###
        elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):
            # remove dollar sign and commas
            s = re.sub('\$|,','', s)
            # convert to float
            value = float(s)
            # return value
            return value
        # otherwise, return NaN
        else:
            return np.nan

    # Apply to the DataFrame
    wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
    
    # Drop null values from budget data
    budget = wiki_movies_df['Budget'].dropna()
    # Turn the data to string to apply regular expressions
    budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)
    # Apply regular expression to clean the data. Then remove any values between a dollar sign and a hyphen (for budgets given in ranges):
    budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
    # Use pattern matches to see budget data
    matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
    matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
    budget[~matches_form_one & ~matches_form_two]
    # Parse budget data, and remove the numbers in [] adding \[\d+\]
    budget = budget.str.replace(r'\[\d+\]\s*', '')
    budget[~matches_form_one & ~matches_form_two]
    # parse the budget values
    wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

    # Parse the release date. Drop null values and convert to string
    release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    # Create date forms
    date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
    date_form_two = r'\d{4}.[01]\d.[123]\d'
    date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
    date_form_four = r'\d{4}'
    # Extract the dates
    release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)
    # Use the built in Pandas method to_datetime() 
    wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)

    # Parse Running Time. Drop null values and convert to string to use regular expressions.
    running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    # Count how many values match the regular expression for running_time
    running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE).sum()
    # Inspect the data that do not match running_time regular expression.
    running_time[running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE) != True]
    # Convert the running_time regular expression to make it more general to except other abbreviations of “minutes” by only searching up to the letter “m.
    running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE).sum()
    # Inspect runnig_time data with the new regular expression.
    running_time[running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE) != True]
    # Extract running_time values
    running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')
    # Extract runnig_time values, and turn the values to numeric values
    running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)
    # Apply a fuction that will convert the data and save the output to the dataframe
    wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)

    # Drop the columns to not need.
    wiki_movies_df.drop('Box office', axis=1, inplace=True)
    wiki_movies_df.drop('Budget', axis=1, inplace=True)
    wiki_movies_df.drop('Running time', axis=1, inplace=True)

    # keep rows where the adult column is False, and then drop the adult column.
    kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')

    # Convert the datatype for video
    kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'

    # Convert the datatype to numeric for budget, id and popularity datas
    kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
    kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
    kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')
    # Convert the release_date to datetime
    kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])

    # Datetime stampt, specify "unix"
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])

    # Find out the index of the outlier row
    movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index
    # Drop the outlier row
    movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)

    # Convert the language data to tuples
    movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)

    # Drop the title_wiki, release_date_wiki, Language, and Production company(s) columns.
    movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)
    # make a function that fills in missing data for a column pair and then drops the redundant column.
    def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
        df[kaggle_column] = df.apply(
            lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
            , axis=1)
        df.drop(columns=wiki_column, inplace=True)

    # run the function for the three column pairs to fill in zeros.
    fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
    fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
    fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')

    # Check the dataframe columns
    for col in movies_df.columns:
        lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
        value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
        num_values = len(value_counts)
        #if num_values == 1:
            #print(col)

    # Drop the video column
    movies_df['video'].value_counts(dropna=False)

    # Rearange the columns
    movies_df = movies_df[['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                        'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                        'genres','original_language','overview','spoken_languages','Country',
                        'production_companies','production_countries','Distributor',
                        'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                        ]]

    # Rename the columns
    movies_df.rename({'id':'kaggle_id',
                    'title_kaggle':'title',
                    'url':'wikipedia_url',
                    'budget_kaggle':'budget',
                    'release_date_kaggle':'release_date',
                    'Country':'country',
                    'Distributor':'distributor',
                    'Producer(s)':'producers',
                    'Director':'director',
                    'Starring':'starring',
                    'Cinematography':'cinematography',
                    'Editor(s)':'editors',
                    'Writer(s)':'writers',
                    'Composer(s)':'composers',
                    'Based on':'based_on'
                    }, axis='columns', inplace=True)

    # use a groupby on the “movieId” and “rating” columns and take the count for each group and ename the “userId” column to “count.”
    # can pivot this data so that movieId is the index
    rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                 .rename({'userId':'count'}, axis=1)                 .pivot(index='movieId',columns='rating', values='count')
    # rename the columns so they’re easier to understand
    rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]
    # Use left merge to merge the movie_df and ratings data
    movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')
    # Fill missing values with 0
    movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)

    # # Make a connection with PostgreSQL
    "postgres://[user]:[password]@[location]:[port]/[database]"
    # # Make a connection with PostgreSQL database
    db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/movie_data"
    # # Create the database engine with the following
    engine = create_engine(db_string)
    # # Import the Movie data to SQL
    movies_df.to_sql(name='movies', con=engine, if_exists='replace', index=False)
    #new_movies = update('movies')
    
    rows_imported = 0
    #get the start_time from time.time()
    start_time = time.time()
    for data in pd.read_csv(f'{file_dir}/ratings.csv', chunksize=1000000):
        print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
        data.to_sql(name='ratings', con=engine, if_exists='append')
        rows_imported += len(data)

        # add elapsed time to final print out
        print(f'Done. {time.time() - start_time} total seconds elapsed')

    print("It is working")



ETL("wikipedia.movies.json", "movies_metadata.csv", "ratings.csv")

#print(kaggle_metadata)[0]
