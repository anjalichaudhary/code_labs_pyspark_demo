## Environment:

- Spark Version: 3.4.0
- Python Version: 3.11

## Read-Only Files:

```
src/app.py
src/tests/*
src/main/__init__.py
src/main/base/*
src/main/job/__init__.py
README.md
requirements.txt
data/*
```

## Requirements:

In this challenge, you are going to write a spark job that does data cleaning. Basically you have to filter `medical.csv` file based on `eligibility.csv` file and do some data manipulation. Sample files are given in `data`.

- `eligibility.csv`
  - it contains the data in the layout `memberId,firstName,lastName`
  - it is a csv file with one line per `memberId`
- `medical.csv`

  - it contains the data in the layout `memberId,fullName,paidAmount`
  - it is a csv file with one line per `memberId`

- `eligibility.csv - medical.csv`
  - Each medical member has one corresponding eligibility record

The project is partially completed and there are 4 methods and a spark session to be implemented in the file `src/main/job/pipeline.py`:

- `init_spark_session(self) -> SparkSession`:

  - create a spark session with master `local` and name `Data Cleaning`

- `filter_medical(self, eligibility: DataFrame, medical: DataFrame) -> DataFrame`:

  - remove all the rows from `medical` whose `memberId` is not present in `eligibility`
  - returned the filtered `medical`

- `generate_full_name(eligibility: DataFrame, medical: DataFrame) -> DataFrame`:

  - `fullName` column in `medical` is empty. So populate it by concatenating `firstName` and `lastName` column from `eligibility`
  - return the `medical`

- `find_max_paid_member(medical: DataFrame) -> str`:

  - find the member which has highest `paidAmount`
  - return the member's `memberId`

- `find_total_paid_amount(medical: DataFrame) -> int`:
  - find the sum of `paidAmount` column in the `medical`
  - return the total sum

Your task is to complete the implementation of that job so that the unit tests pass while running the tests. You can use the give tests check your progress while solving problem.

## Commands

- run:

```bash
source venv/bin/activate; python3 src/app.py data/eligibility.csv data/medical.csv
```

- install:

```bash
bash make.sh __install; source venv/bin/activate; pip3 install -r requirements.txt
```

- test:

```bash
source venv/bin/activate; py.test -p no:warnings
```
