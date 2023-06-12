# This is a sample Python script.
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import json
import plotly.figure_factory as ff
import datetime
import time
import copy

def data_from_json():
    # Opening JSON file
    f = open('data.json')

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # Iterating through the json
    # list
    for i in data['Flow Sequence']:
        print(i)

    # Closing file
    f.close()
    return data


def json_to_df(json_data):
    """ convert json into excel """
    dict_data = {}
    for key in json_data.keys():
        dict_data[key] = pd.DataFrame(json_data.get(key)).T
    return dict_data

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def plot_new_gantt_chart(num_links, num_flows, sequence_best, process_time):
    f_record = {}
    for i in range(num_links):  # each flow
        for j in range(num_flows):
            print(sequence_best[i])
            processing_time_for_flow = int(process_time.values[i][j])
            start_time_seconds =sequence_best[i][j]
            #if (start_time_seconds!=0):
            start_time = str(datetime.timedelta(seconds=start_time_seconds))  # convert seconds to hours, minutes and seconds
            end_time = str(datetime.timedelta(seconds=start_time_seconds+processing_time_for_flow))

            f_record[(i, j)] = [start_time, end_time]

    # print("J Record",j_record)
    print("f_record:", f_record)
    df = []
    for i in range(num_links):
        for j in range(num_flows):
            # if ( m!=3 & j!=0):
            # print(j_record[j, m])
            #flow = flows[flow_key]
            df.append(
                dict(Task='Link %s' % (i), Start='2020-02-01 %s' % (str(f_record[(j, i)][0])), \
                     Finish='2020-02-01 %s' % (str(f_record[(j, i)][1])),
                     Resource='Flow %s '%(j)))

    df_ = pd.DataFrame(df)
    df_.Start = pd.to_datetime(df_['Start'])
    df_.Finish = pd.to_datetime(df_['Finish'])
    start = df_.Start.min()
    end = df_.Finish.max()

    df_.Start = df_.Start.apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S'))
    df_.Finish = df_.Finish.apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S'))
    print("Df", df)
    data = df_.to_dict(orient='records')

    final_data = {
        'start': start.strftime('%Y-%m-%dT%H:%M:%S'),
        'end': end.strftime('%Y-%m-%dT%H:%M:%S'),
        'data': data}

    fig = ff.create_gantt(df, index_col='Resource', show_colorbar=True, group_tasks=True, showgrid_x=True,
                          title='Traffic Schedule')
    fig.show()
    return final_data, df

def plotSchedule():
    data = data_from_json()
    data_json = json_to_df(data)
    flow_sequence_tmp = data_json['Flow Sequence']
    process_time_tmp = data_json['Processing Time']
    with open('solution.npy', 'rb') as f:
    	best_sequence = np.load(f)
    print(best_sequence) #no commas issue. How do we now do it.
    #best_sequence = [[ 0, 40, 70, 80, 100, 140], [ 0, 20, 30, 50, 60, 90], [ 0, 30, 40, 60, 100, 120],
    #                [ 10, 60, 80, 100, 120, 140], [ 0, 20, 40, 50, 70, 80], [ 0, 40, 60, 70, 90, 100]]
    final_data, df = plot_new_gantt_chart(6, 6, best_sequence, process_time_tmp)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    plotSchedule()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
