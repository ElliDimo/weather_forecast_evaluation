from functions import *
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    station_df = read_data('input/station_data.parquet')
    forecast_df = read_data('input/forecast_data.parquet')

    metrics = ['temperature', 'windspeed', 'winddirection']
    sources = ['GFS', 'IFS', 'ICONEU']

    # calculate average wind speed & direction values
    station_df_hourly = average_wind(station_df)

    # merge with forecast data
    df = station_df_hourly.merge(forecast_df, how='left', on='datetime')
    df = df.rename(columns={"wind_speed": "windspeed", "wind_direction": "winddirection"})
    df.to_csv('merged_df.csv')
    accuracy_df = calculate_accuracy(df, metrics, sources)

    print(f'--------------------- Weather Forecast Evaluation ---------------------')
    print(f'* Error metrics used: RMSE and MAE')
    print(f'* Temperature unit: degrees Celsius')
    print(f'* Wind speed unit: m/s')
    print(f'* Wind direction unit: degrees')
    print()

    for metric in metrics:
        print(f'--------------------- {(" " * 1).join(metric)} ---------------------')
        print(accuracy_df.xs(metric, axis=1))
        print()

    plot_metrics(df, 'part_of_day', metrics, sources)

    plot_box_plots(df, metrics, sources)


