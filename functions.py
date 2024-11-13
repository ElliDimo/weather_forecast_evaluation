import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from bokeh.plotting import figure, show
from bokeh.io import export_svg
from bokeh.models import ColumnDataSource, Whisker
from bokeh.transform import factor_cmap
from bokeh.models import Range1d
from bokeh.palettes import Spectral5
from math import sin, cos, radians, degrees, atan2


def read_data(file_path):
    df = pd.read_parquet(file_path)
    df['datetime'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S').dt.tz_localize('UTC')
    return df


def average_wind(station_df):
    # Calculating x (cos) & y (sin) of unit vector of wind direction degrees
    station_df['sinx'] = station_df['wind_direction'].apply(lambda x: sin(radians(x)))
    station_df['cosx'] = station_df['wind_direction'].apply(lambda x: cos(radians(x)))

    # Aggregating wind speed to average and sin, cos to sum
    station_df_hourly = station_df.set_index(station_df['datetime'], drop=False). \
        groupby(pd.Grouper(freq='1h', origin='end_day', closed='left')).agg({'wind_speed': ['mean'],
                                                                             'sinx': ['sum'],
                                                                             'cosx': ['sum']}).reset_index()
    station_df_hourly.columns = ['datetime', 'wind_speed', 'sinx', 'cosx']

    # calculating average angle from atan2 of x, y
    station_df_hourly['wind_direction'] = station_df_hourly[['sinx', 'cosx']]. \
        apply(lambda x: degrees(atan2(x['sinx'], x['cosx'])), axis=1)

    # convert to 0-360 range
    station_df_hourly['wind_direction'] %= 360

    # merge with point on time temperature and keep hours only
    station_df_hourly = station_df_hourly.merge(station_df[['datetime', 'temperature']], how='left', on='datetime')
    return station_df_hourly


def calculate_accuracy(df, metrics, sources):
    cmidx = pd.MultiIndex.from_product([metrics, sources],
                                       names=['metric', 'source'])
    accuracy_df = pd.DataFrame(index=['RMS', 'MAE'], columns=cmidx)

    for metric in metrics:
        for source in sources:
            forecast_column = metric + '_forecast_' + source
            df_not_nas = df.dropna(subset=[metric, forecast_column])

            rms = mean_squared_error(df_not_nas[metric], df_not_nas[forecast_column], squared=False)
            mae = mean_absolute_error(df_not_nas[metric], df_not_nas[forecast_column])
            accuracy_df.loc['RMS', (metric, source)] = rms
            accuracy_df.loc['MAE', (metric, source)] = mae

    accuracy_df.to_csv('accuracy_df.csv')
    return accuracy_df


def map_part_of_day(df):
    df['part_of_day'] = np.nan
    df['part_of_day'][df['datetime'].dt.hour.between(0, 6, inclusive='left')] = 'night'
    df['part_of_day'][df['datetime'].dt.hour.between(6, 12, inclusive='left')] = 'morning'
    df['part_of_day'][df['datetime'].dt.hour.between(12, 20, inclusive='left')] = 'afternoon'
    df['part_of_day'][df['datetime'].dt.hour.between(20, 24, inclusive='left')] = 'evening'

    return df


def plot_metrics(df, group, metrics, sources):
    colors = ['blue', 'red', 'green']

    for metric in metrics:
        p = figure(title="Hourly " + metric, x_axis_label='time', y_axis_label=metric, x_axis_type='datetime', plot_width=1000, plot_height=500)
        p.line(df.datetime.to_list(), df[metric].to_list(), legend_label="Actual " + metric, color="black", line_width=2)

        for i, source in enumerate(sources):
            forecast_column = metric + "_forecast_" + source
            p.line(df.datetime.to_list(), df[forecast_column].to_list(), legend_label=source, color=colors[i], line_width=2)
        p.add_layout(p.legend[0], 'above')
        p.legend.orientation = "horizontal"
        p.legend.location = "center"
        export_svg(p, filename="plots//" + metric + "_actual_vs_forecast.svg")

        forecast_columns = [metric + '_forecast_' + source for source in sources]
        df_melted = pd.melt(df, id_vars=['datetime', metric], value_vars=forecast_columns)
        df_melted['variable'] = df_melted['variable'].str.replace(str(metric + '_forecast_'), '')
        df_melted = df_melted.rename(columns={"variable": "source", "value": "forecast"})
        df_melted_not_nas = df_melted.dropna(subset=[metric, 'forecast'])
        df_melted_not_nas = map_part_of_day(df_melted_not_nas)

        # Per group figures
        accuracy_df_grouped = df_melted_not_nas.groupby(['source', group]).\
            apply(lambda x: pd.Series(dict(rmse=mean_squared_error(x[metric], x['forecast'], squared=False), mae=mean_absolute_error(x[metric], x['forecast'])))).reset_index()

        groups = accuracy_df_grouped[group].unique()
        accuracy_df_grouped = accuracy_df_grouped.sort_values(by=['source', group])

        p = figure(title=f"RMSE of {metric} per source & {group}", x_axis_label=group, y_axis_label='RMSE', x_range=groups, plot_width=1000, plot_height=500)
        for i, source in enumerate(sources):
            p.line(accuracy_df_grouped[accuracy_df_grouped['source'] == source][group].to_list(), accuracy_df_grouped[accuracy_df_grouped['source'] == source].rmse.to_list(), legend_label=source,
                   color=colors[i], line_width=2)

        p.add_layout(p.legend[0], 'above')
        p.legend.orientation = "horizontal"
        p.legend.location = "center"
        export_svg(p, filename = "plots//" + metric + "_accuracy_per_" + group + ".svg")

        # Per day figures
        accuracy_df_grouped = df_melted_not_nas.set_index(df_melted_not_nas['datetime'], drop=False). \
                        groupby([pd.Grouper(freq='1d'),'source']).apply(lambda x: pd.Series(dict(rmse=mean_squared_error(x[metric], x['forecast'], squared=False),
                                                                                                 mae=mean_absolute_error(x[metric], x['forecast'])))).reset_index()

        p = figure(title=f"RMSE of {metric} per source & day", x_axis_label='date', y_axis_label='RMSE',x_axis_type='datetime', plot_width=1000, plot_height=500)
        for i, source in enumerate(sources):
            p.line(accuracy_df_grouped[accuracy_df_grouped['source']==source].datetime.to_list(), accuracy_df_grouped[accuracy_df_grouped['source']==source].rmse.to_list(), legend_label=source, color=colors[i], line_width=2)

        p.add_layout(p.legend[0], 'above')
        p.legend.orientation = "horizontal"
        p.legend.location = "center"
        export_svg(p, filename = "plots//" + metric + "_accuracy_per_date.svg")

    return


def plot_box_plots(df, metrics, sources):
    for metric in metrics:
        forecast_columns = [metric + '_forecast_' + source for source in sources]
        df_melted = pd.melt(df, id_vars=['datetime', metric], value_vars=forecast_columns)
        df_melted['variable'] = df_melted['variable'].str.replace(str(metric + '_forecast_'), '')
        df_melted = df_melted.rename(columns={"variable": "source", "value": "forecast"})

        kinds = df_melted.source.unique()
        df_melted['diff'] = df_melted[metric] - df_melted['forecast']

        # compute quantiles
        qs = df_melted.groupby("source")['diff'].quantile([0.25, 0.5, 0.75])
        qs = qs.unstack().reset_index()
        qs.columns = ["source", "q1", "q2", "q3"]

        # compute IQR outlier bounds
        iqr = qs.q3 - qs.q1
        qs["upper"] = qs.q3 + 1.5 * iqr
        qs["lower"] = qs.q1 - 1.5 * iqr
        df_melted = pd.merge(df_melted, qs, on="source", how="left")

        source = ColumnDataSource(qs)

        p = figure(x_range=kinds, tools="", toolbar_location=None,
                   title=f'{metric} error distribution by forecast source',
                   background_fill_color="#eaefef", y_axis_label="Error")

        # outlier range
        whisker = Whisker(base="source", upper="upper", lower="lower", source=source)
        whisker.upper_head.size = whisker.lower_head.size = 30
        p.add_layout(whisker)

        # quantile boxes
        cmap = factor_cmap("source", palette=Spectral5, factors=sorted(df_melted.source.unique()))
        p.vbar("source", 0.7, "q2", "q3", source=source, color=cmap, line_color="black")
        p.vbar("source", 0.7, "q1", "q2", source=source, color=cmap, line_color="black")

        # outliers
        outliers = df_melted[~df_melted['diff'].between(df_melted.lower, df_melted.upper)]
        p.scatter("source", "diff", source=outliers, size=6, color="black", alpha=0.3)

        p.xgrid.grid_line_color = None
        p.axis.major_label_text_font_size = "14px"
        p.axis.axis_label_text_font_size = "12px"

        range_min = df_melted[['diff', 'lower']].min().min()
        range_min = range_min - 0.25 * abs(range_min)
        range_max = df_melted[['diff', 'upper']].max().max()
        range_max = range_max + 0.25 * abs(range_min)
        p.y_range = Range1d(range_min, range_max)

        export_svg(p, filename="plots//" + metric + "_diff_box_plot.svg")

    return
