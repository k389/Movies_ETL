# Movies_ETL
ETL, Python, SQL

# Project Overview/Challenge

- Create an authomated ETL pipeline
- Extract data from multiple sources
- Clean and transform the data authomatically using Pandas and regular expressions
- Load new data into PostreSQL

# Resources

- Data Source: wikipedia.movies.json, movies_metadata.csv, ratings.csv (the files are too large to import to GitHub)
- Software: Python Python 3.6.9:: Anaconda, Inc., Jupyter Notebook, 6.0.2, Visual Studio Code, 1.40.2., PgAdmin 4

# Summary
- Used Visual Studio Code to create an automated pipeline.
- Created a function to import data files using pandas
	- import data from wikipedia json file (wikipedia.movies.json)
	- import data from Kaggle metadata (movies_metadata.csv)
	- import data from MovieLens rating data (ratings.csv)
- Clean and transform the datas to be able to perform analysis and create a dataframe
- Load the cleaned and trnascformed dataframe to SQL database
- Perform data transformation based on document assumption.
	- 1. Add try-exempt blocks to the data to account imdb.id character count (extracting only ids with 7 characters). 
	- 2. Assume the imported wiki file needs to be cleaned. 
		- rename the columns names to make them more readable
		- clean the wiki data leaving only the original movie title
		- transform the wiki data to a dataframe
		- remove columns that are missing 90% of the data. The columns that are missing the data cannot be used for analysis
	- 3. Transform the column datatypes to match the data in the row
		- transform the box_office column data using regular expressions
		- change the box_office datatype to numeric
		- transform the budget data column using regular expressions
		- change the release_date to datetime datatype using regular expressions
		- change the running_time to reflect the correct datatype.
		- After transformation extract box_office, budget, release_date and running_time data to new columns in the DataFrame
			- drop the original columns
	- 4. Assume the kaggle_metadata columns do not have correct datatypes
		- Change the budget, id, popularity column datas to numeric values
		- Change the release_date and to datetime values using pandas
 	- 5. When merging the Wikipedia and Kaggle dataframes assume that several columns will be duplicates
		- inspect the merged dataframe to remove the duplicate columns
		- before dropping the duplicate columns, make sure that the columns kept have the complete data for the analysis
- The automated pipeline
	- will import the data from the json and csv datas
	- perform the nesssecity transformations to Wikipedia and Kaggle dataframes
	- Merge the dataframes into one new dataframe
	- Remove the duplicate columns from the merged dataframe
	- Import the new dataframe to SQL database
		