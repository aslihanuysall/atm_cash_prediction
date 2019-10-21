import pandas as pd
import model
import config
import utils
import calendar
import sys
import traceback

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print('no argument sent')
        n_days_to_feed_model = 90
        n_days_to_predict = 30
        n_features = 8
    else:
        n_days_to_feed_model = sys.argv[1]
        n_days_to_predict = sys.argv[2]
        n_features = sys.argv[3]
    try:

        ## Get Raw Data
        df = pd.read_csv('/Users/aslihanuysal/Desktop/ds_exercise_data.csv')
        df = df.fillna(0)

        ## Data Manipualtions & Adding New Features
        df['DateTime'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='DateTime', ascending=True)

        df["Weekday"] = df["DateTime"].apply(lambda x: calendar.day_name[x.weekday()])
        df["DayOfMonth"] = df["DateTime"].apply(lambda x: x.day)
        df["Month"] = df["DateTime"].apply(lambda x: x.month)
        df["MonthName"] = df["DateTime"].apply(lambda x: calendar.month_name[x.month])
        df["Year"] = df["DateTime"].apply(lambda x: x.year)
        df["WeekOfMonth"] = df["DateTime"].apply(lambda x: config.week_of_month(x))
        df["yearlyDeviation"] = df["Year"].apply(lambda x: config.yearlyDeviationDict[x]["ratio"])
        df["yearlyDeviationChange"] = df["Year"].apply(lambda x: config.yearlyDeviationDict[x]["changeRatio"])
        # Working Day Or Not
        df["workingDayOrNot"] = df["Weekday"].apply(lambda x: True if x in config.weekdaysIndexList else False)
        df["Season"] = df["Month"].apply(lambda x: utils.getSeason(x))

        categoricalFeatureNameList = list(df.select_dtypes(exclude=["number","datetime"]))
        categoricalFeatureIndexList = [df.columns.get_loc(c) for c in categoricalFeatureNameList]

        scaledArray = model.preprocessModelDf(df, categoricalFeatureIndexList ).values
        model, history = model.runModel(scaledArray, n_features, n_days_to_feed_model, n_days_to_predict)

        model.plotLossHistory(history)
        rmse = model.makePrediction(test_X,test_y,model,n_days_to_feed_model,n_features)


    except Exception as e:
        errorLog = str(traceback.format_exc())
        print(errorLog)









