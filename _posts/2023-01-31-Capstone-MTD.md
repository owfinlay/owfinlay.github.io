---
layout: post
title: "Capstone Metadata Generator"
date: 2023-01-31 10:56 -0500
categories: GA NLP
---

This is the last Python script I wrote for the project. It's to collect data about my already-generated bins of reviews.

First I made the essential imports for establishing metadata about the reviews. Polars for manipulating dataframes, OS for navigating files, and RE for working with regular expressions.

```Python
import polars as pl
import os
import re
  
path = os.path.abspath('E:/(E)General Assembly stuff/Capstone/Final')
filename = 'cap_reviews.csv'
  
filepath = os.path.join(path, filename)
  
df = pl.read_csv(filepath)
```

The reviews were stored in a CSV with a column representing each bin of data--so each year's data was split into two different columns of either negative or positive reviews. I firstly wanted to combine these columns so each represented *all* of a given year's data; and to do this I chose to use Polars's [select](https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.select.html#polars.DataFrame.select) and [vstack](https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.vstack.html#polars.DataFrame.vstack) functions.

Select allows you to chose certain columns of a dataframe and turn them into a dataframe; vstack allows you to stack these dataframes vertically. 

Columns are named like this: "{year}/{mode}/{sentiment}", so slicing a column name with `<column_name>[:4]` will return just the year. And since, after that slicing, columns are named *just* for their year, vstack works like a charm.

```Python
better_df = df.select([
    
    pl.col(i).alias(i[:4])            # Selecting even columns #
    for i in df.columns[::2] #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    ]).vstack(df.select([            # Stacking odds onto the end of evens #
        
        pl.col(i).alias(i[:4])         # Selecting odd columns #
        for i in df.columns[1::2] #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        ]))
```

Next I started to use regular expression, or Regex, patterns to record data about the reviews. These are the Regex patterns I used.

```Python
 #Regex to identify HTML elements like <br></br>
html_pattern = "<(?:\"[^\"]*\"['\"]*|'[^']*'['\"]*|[^'\">])+>"
	#(got this from https://uibakery.io/regex-library/html-regex-python)

alph_pattern = re.compile(r"[A-Za-z]")         # Alphabetical #

upper_pattern = re.compile(r"[A-Z]")        # Uppercase Alphabetical #
```

I'm not good enough with Polars yet to be able to collect this data within a dataframe, so I just did a nested dictionary that I would later convert to a dataframe. 

I needed to change every column to either a Polars Series or a continuous string, so I first made variables representing those things, `col_Ser` and `col_as_string` respectively. Then I collected a number of statistics for each bin. 

```Python
mtd = {}

for i in df.columns:

    col_Ser = df.select(i).get_columns()[0]
    col_as_string = col_Ser.str.concat("")[0]
    
    mtd[i] = {
        'path'          : i,
        'is_not_full'   : col_Ser.is_null().any(),
        'mean_len'      : col_Ser.str.lengths().mean(),
        'median_len'    : col_Ser.str.lengths().median(),
        'std_len'       : col_Ser.str.lengths().std(),
        'html_per_K'    : len(col_Ser.str.concat("")
					        .str.count_match(html_pattern))/25,
        'pct_upper'     : len(re.findall(upper_pattern, col_as_string)) \
            / len(re.findall(alph_pattern, col_as_string)),
        'pct_nums'      : len(re.findall('[0-9]', col_as_string)) \
            / len(col_as_string),
        'pct_!alnumeric': len(re.findall('\W', col_as_string)) / len(col_as_string)
        }
```

Now I just needed to convert this to a dataframe (which is a breeze with `pl.from_dicts`) (I spent two hours trying to figure how to do this with `pl.from_dict` before realizing `pl.from_dicts` was a thing) (Don't be like me).

I changed the working directory to my capstone folder so I could write there more easily. Then a quick list comprehension, and voila! I also wrote the reviews per year to a CSV so I could further analyze it in Excel.

```Python
os.chdir("E:/(E)General Assembly stuff/Capstone")

mtd_df = pl.from_dicts([mtd[i] for i in mtd.keys()])
mtd_df.write_csv("cap_reviews_metadata")

better_df.write_csv("yearly_reviews")
```


I also created a CSV with each year's most common words. I made two versions of this CSV, one with stop-words (common, inconsequential ones like "the", "is", and "or"). Here is the code for that part:

```Python
  # NLTK stands for Natural Language Toolkit. 
  # I'm importing its collection of stopwords
from nltk.corpus import stopwords

  # Initializing a list to fill it with dataframes, sized (2,100), for each year
most_common_words = []
  # Naming the set of English language stopwords
stop_words = set(stopwords.words('english'))

  # For each year, turn the reviews into a list of individual words their counts
for i in better_df.columns:
    col_Ser = better_df.select(i).get_columns()[0]
    col_as_string = col_Ser.str.concat(" ")
    
    most_common_words.append(col_as_string.str.split(by=" ")[0]\
        .value_counts(sort=True).head(100))
    most_common_words[-1].columns = [i+'_SW', i+'_SW_ct']
    
    
    most_common_words.append(col_as_string.str.split(by=" ")[0]\
        .value_counts(sort=True).filter(
            ~pl.col(i).str.to_lowercase().is_in(list(stop_words)+["","null"])
            ).head(100))
    most_common_words[-1].columns = [i+"_no_SW", i+"_no_SW_ct"]


    # Going from a list of DFs -> one big DF
mcw = pl.DataFrame(range(1, 101)).hstack([i.get_columns()[j]            
                                          for i in most_common_words
                                          for j in [0,1]])

    # Exporting the DF as a CSV file
mcw.write_csv("cap_most_common_words")
```

... and that's it for the coding in my final project for General Assembly!