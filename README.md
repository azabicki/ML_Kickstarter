# Workflow

## Process raw data to obtain a more handable csv file

In a first step, I downloaded all zip files from the [**Web Robots Kickstarter**](https://webrobots.io/kickstarter-datasets/) page. These files are a) in some cases too big to be uploaded directly to GITHUB, and b) also very redundant containing a lot of duplicated entries. 

Therefor i extracted only the data from the zip files which is used later in the main analysis notebook. This procedure, loading the data file by file and deleting immediately the _features of no interest_ and the _duplicats_, also has the advantage that the dataframe does not gets to big and consumes a lot of memory. 

Alltogether, the **_58 zip archives_** contain **_1814 csv_** files, which in the end deliver **_6,595,455 instances_** of Kickstarter Projects.

After removing duplicates (by 'id') only **_319,639 projects_** left, showing how reduntant all the zip files are. Having a total of 48 features, the saved csv file is 1.74GB big, and zipped still 290MB. Therefor I split up the dataframe into 

After unzipping and deleting duplicates in each loading iteration,  