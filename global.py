
# Import to be able to run the code below

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import shapefile as shp
from pyproj import Proj, transform
import cv2
import mysql.connector


def shp_to_df(sf):

    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shps = [s.points for s in sf.shapes()]
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)

    return df


def calculate_centroid(points):

    x_sum = 0
    y_sum = 0
    for coords in points:
        x_sum += coords[0]
        y_sum += coords[1]

    x_mean = x_sum/len(points)
    y_mean = y_sum/len(points)

    return x_mean, y_mean


def calculate_boundaries(points):

    x_max = -99999999
    x_min = 99999999
    y_max = -99999999
    y_min = 99999999

    for coords in points:
        if coords[0] > x_max:
            x_max = coords[0]
        if coords[0] < x_min:
            x_min = coords[0]
        if coords[1] > y_max:
            y_max = coords[1]
        if coords[1] < y_min:
            y_min = coords[1]

    max_bound = (x_max, y_max)
    min_bound = (x_min, y_min)

    return max_bound, min_bound


def process_shape_boundaries(df_sf, sf):

    shape_dict = {}
    index_list = df_sf.index.tolist()

    for zone_id in index_list:
        # for each zone id available in the shapefile
        if zone_id not in shape_dict:
            # we only process the coordinates if it is not yet included in the dictionary
            shape_zone = sf.shape(zone_id)

            points = [(i[0], i[1]) for i in shape_zone.points]

            x_center, y_center = calculate_centroid(points)
            max_bound, min_bound = calculate_boundaries(points)

            # we add to the dictionary, for the zone id, the shape boundaries as well
            # as the coordinates of the center of the shape znd the zone extreme boundaries
            shape_dict[zone_id] = {}
            shape_dict[zone_id]['points'] = points
            shape_dict[zone_id]['center'] = (x_center, y_center)
            shape_dict[zone_id]['max_bound'] = max_bound
            shape_dict[zone_id]['min_bound'] = min_bound

        return shape_dict

def find_max_coords(shape_dict):

    all_max_bound = []
    all_min_bound = []

    for zone in shape_dict:
        zone_shape = shape_dict[zone]
        max_bound_zone = zone_shape['max_bound']
        min_bound_zone = zone_shape['min_bound']
        all_max_bound.append(max_bound_zone)
        all_min_bound.append(min_bound_zone)

    map_max_bound, unused_max = calculate_boundaries(all_max_bound)
    unused_min, map_min_bound = calculate_boundaries(all_min_bound)

    return map_max_bound, map_min_bound


def define_projection(map_max_bound, map_min_bound, image_size):

    # We get the max 'coordinates' for both the target image and the shape we want to draw
    image_x_max = image_size[0]
    image_y_max = image_size[1]
    map_x_max = map_max_bound[0]
    map_y_max = map_max_bound[1]
    map_x_min = map_min_bound[0]
    map_y_min = map_min_bound[1]

    projection = {}

    # we check which size is bigger to know based on which axis we want to scale our shape to
    # we do the comparison using the aspect ratio expectations (dividing each axis by the
    # size of the target axis in the new scale)
    if (map_x_max - map_x_min)/image_x_max > (map_y_max - map_y_min)/image_y_max:
        conversion = image_x_max / (map_x_max - map_x_min)
        axis_to_center = 'y'  # we store the axis we will want to center on based on which
        # axis we perform the scaling from
    else:
        conversion = image_y_max / (map_y_max - map_y_min)
        axis_to_center = 'x'

    projection['image_size'] = image_size
    projection['map_max_bound'] = map_max_bound
    projection['map_min_bound'] = map_min_bound
    projection['conversion'] = conversion
    projection['axis_to_center'] = axis_to_center

    return projection


def convert_projection(x, y, projection, inverse=False):

    x_min = projection['map_min_bound'][0]
    y_min = projection['map_min_bound'][1]
    conversion = projection['conversion']

    if inverse is False:
        # to be able to center the image, we first translate the coordinates to the origin
        x = (x - x_min) * conversion
        y = (y - y_min) * conversion
    else:
        x = (x + x_min) / conversion
        y = (y + y_min) / conversion

    return x, y


def convert_shape_boundaries(zone_shape_dict, projection):

    converted_dict = {}
    axis_to_center = projection['axis_to_center']
    image_x_max = projection['image_size'][0]
    image_y_max = projection['image_size'][1]
    map_max_bound_converted = (convert_projection(projection['map_max_bound'][0], projection['map_max_bound'][1], projection))
    map_min_bound_converted = (convert_projection(projection['map_min_bound'][0], projection['map_min_bound'][1], projection))

    if axis_to_center == 'x':
        center_translation = (image_x_max - (map_max_bound_converted[0] - map_min_bound_converted[0]))/2
    else:
        center_translation = (image_y_max - (map_max_bound_converted[1] - map_min_bound_converted[1]))/2

    for zone_id in zone_shape_dict:
        curr_shape = zone_shape_dict[zone_id]

        points = curr_shape['points']
        x_center = curr_shape['center'][0]
        y_center = curr_shape['center'][1]
        max_bound = curr_shape['max_bound']
        min_bound = curr_shape['min_bound']

        converted_points = []
        for point in points:
            # we convert the coordinates to the new coordinate system
            converted_point = [0, 0]
            converted_point[0], converted_point[1] = convert_projection(point[0], point[1], projection)
            # we center the map on the axis that was not used to scale the image
            if axis_to_center == 'x':
                converted_point[0] = converted_point[0] + center_translation
            else:
                converted_point[1] = converted_point[1] + center_translation

            # we mirror the image to match the axis alignment
            converted_point[1] = image_y_max - converted_point[1]
            converted_points.append(converted_point)

        # we convert the center and the max and min boundaries
        x_center, y_center = calculate_centroid(converted_points)
        max_bound = (convert_projection(max_bound[0], max_bound[1], projection))
        min_bound = (convert_projection(min_bound[0], min_bound[1], projection))

        # We edit the dictionary with the new coordinates
        converted_dict[zone_id] = {}
        converted_dict[zone_id]['points'] = converted_points
        converted_dict[zone_id]['center'] = (x_center, y_center)
        converted_dict[zone_id]['max_bound'] = max_bound
        converted_dict[zone_id]['min_bound'] = min_bound

    return converted_dict


def get_shape_set_to_draw(map_type, shape_dict, df_sf, image_size):

    # we define if we want to draw the whole map or only a borough (in this case map_type
    # should be the borough name)
    if map_type == 'total':
        shape_dict = shape_dict
    else:
        # we select the list of zone_id we want to draw that belong only to the targeted
        # borough to draw
        shape_dict = reduce_shape_dict_to_borough(shape_dict, df_sf, map_type)

    # We define the projection parameters to be able to convert the coordinates into
    # the image scale coordinate system
    # we convert the coordinates of the shapes to draw
    map_max_bound, map_min_bound = find_max_coords(shape_dict)
    projection = define_projection(map_max_bound, map_min_bound, image_size)
    converted_shape_dict = convert_shape_boundaries(shape_dict, projection)

    return converted_shape_dict, projection


def reduce_shape_dict_to_borough(shape_dict, df_sf, borough_name):

    borough_df = df_sf[df_sf['borough'] == borough_name]
    borough_id = []
    for objectid in borough_df.index:
        borough_id.append(objectid)

    reduced_shape_dict = {}
    # we add to the reduced_shape_dict only the zones belonging to the borough area targeted
    for zone_id in borough_id:
        reduced_shape_dict[zone_id] = shape_dict[zone_id]

    return reduced_shape_dict


def draw_base_map(draw_dict):

    # We extract the variables we will need from the input dictionary
    image_size = draw_dict['image_size']
    map_type = draw_dict['map_type']
    title = draw_dict['title']
    shape_dict = draw_dict['shape_dict']
    df_sf = draw_dict['df_sf']
    render_single_borough = draw_dict['render_single_borough']

    # first we create a blank image, on which we will draw the base map
    width = image_size[0]
    height = image_size[1]
    base_map = np.zeros((height, width, 3), np.uint8)  # Size of the image 1080 height, 1920 width, 3 channels of colour
    base_map[:, :] = [0, 0, 0]  # Sets the color to white

    # we isolate the set of shapes we want to draw in the right coordinate system
    converted_shape_dict, projection = get_shape_set_to_draw(map_type, shape_dict, df_sf, image_size)

    if render_single_borough is False:
        # we use the projection parameters from the borough we want to focus on
        # we calculate the coordinates for the whole map
        converted_shape_dict = convert_shape_boundaries(shape_dict, projection)

    # we draw each shape of the dictionary on the blank image,
    # either the full map or only a borough
    for item in converted_shape_dict:
        shape = converted_shape_dict[item]
        points = shape['points']
        pts = np.array(points, np.int32)
        cv2.polylines(base_map, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)

    # we display general text information
    display_general_information_text(base_map, map_type, title)

    return base_map, projection





def interpolate_next_position(origin_coords, destination_coords, tot_frames, curr_frame):

    # as to perform the arithmetic operations, we convert everything to float for more
    # precision
    x_origin = float(origin_coords[0])
    y_origin = float(origin_coords[1])
    x_destination = float(destination_coords[0])
    y_destination = float(destination_coords[1])
    tot_frames = float(tot_frames - 1)
    curr_frame = float(curr_frame)

    delta_x = (x_destination - x_origin)/tot_frames
    delta_y = (y_destination - y_origin)/tot_frames

    # the rendering with OpenCV demands integers values for the positioning, so we convert
    # w and y to int
    new_x = int(x_origin+delta_x*curr_frame)
    new_y = int(y_origin+delta_y*curr_frame)

    return new_x, new_y


def render_point_on_map(x_point, y_point, weight, base_map, colour):

    cv2.circle(base_map, (x_point, y_point), weight, colour, -1)


def convert_id_shape(idx, inverse=False):

    if inverse is False:
        idx = idx - 1
    else:
        idx = idx + 1

    return idx


def build_query_dict(render_animation_dict):

    # First, we extract the variables we will need from the input dictionary
    time_granularity = render_animation_dict['time_granularity']
    filter_query_on_borough = render_animation_dict['filter_query_on_borough']
    weekdays = render_animation_dict['weekdays']

    # we instantiate the query_dict and start filling it with query parameters
    query_dict = {}
    query_dict['data_table'] = render_animation_dict['data_table']
    query_dict['lookup_table'] = render_animation_dict['lookup_table']
    query_dict['aggregated_result'] = render_animation_dict['aggregated_result']
    query_dict['aggregate_period'] = render_animation_dict['aggregate_period']
    query_dict['weekdays'] = weekdays

    # we handle the borough related WHEN statement
    if filter_query_on_borough is False:
        query_dict['filter_query_on_borough'] = False
    else:
        query_dict['filter_query_on_borough'] = filter_query_on_borough

    # we handle the time related WHEN statements
    period = render_animation_dict['period']
    start_date = period[0]
    end_date = period[1]

    if start_date == end_date:
        query_dict['date'] = start_date

    else:
        # if the period is more than one date, we will have to loop through the
        # date range and render multiple series of 60 frames (1 second at 60 fps per day)
        # Thus the loop needs to be handled by the main plotting function, and here we
        # simply add a flag to the query dict that will be transformed by the plotting
        # function
        query_dict['date'] = 'loop_through_period'

    # used specifically for the animation logic
    if time_granularity == 'specific_weekdays' or weekdays != ():
        specific_weekdays = render_animation_dict['weekdays']
        query_dict['specific_weekdays'] = 'on_specific_weekdays'

    # used specifically for the animation logic
    elif time_granularity == 'period':
        query_dict['specific_weekdays'] = False

    # used specifically for the heat_map logic
    elif time_granularity == 'weekdays_vs_weekends':
        query_dict['specific_weekdays'] = 'weekdays_vs_weekends'

    return query_dict


def compute_min_max_passengers(trips_list, idx_weight):

    min_passenger_itinerary = min(trips_list, key=lambda x: x[idx_weight])
    max_passenger_itinerary = max(trips_list, key=lambda x: x[idx_weight])
    max_passenger = max_passenger_itinerary[idx_weight]
    min_passenger = min_passenger_itinerary[idx_weight]

    return min_passenger, max_passenger


def compute_weight(map_type, weight, max_passenger):
    # we normalise the weight of the point based on the max number of passengers
    # which means that from one day to another, although the biggest point will have the
    # same size, it will not represent the same number of passengers (compromise to
    # prevent having huge differences between the points, or squishing too much the scale
    # by using a log).

    if map_type != 'total':
        weight = weight/max_passenger*30
    else:
        weight = weight/max_passenger*20

    weight = int(weight)

    return weight


def display_specific_text_animation(rendered_frame, date_info, map_type, min_pass, max_pass):
    # note that these position are based on an image size of [1920, 1080]
    font = cv2.FONT_HERSHEY_SIMPLEX
    agg_per = date_info[0]
    date = date_info[1]

    # displays the date and the weekday, and if it is a special date
    if agg_per is False:
        cv2.putText(rendered_frame, date, (40, 150), font, 1.3, (221, 221, 221), 1, cv2.LINE_AA)

        special_dates_2018 = {'2018-01-01': 'New Year', '2018-12-25': 'Christmas', '2018-02-14': 'Valentine\'s Day', '2018-07-04': 'National Day', '2018-07-01': 'Hottest Day', '2018-01-07': 'Coldest Day'}
        if date in special_dates_2018:
            cv2.putText(rendered_frame, special_dates_2018[date], (40, 200), font, 1.3, (221, 221, 221), 1, cv2.LINE_AA)

        date_timestamp = pd.Timestamp(date)
        weekday = date_timestamp.dayofweek
        weekdays = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        weekday = weekdays[weekday]
        cv2.putText(rendered_frame, weekday, (40, 95), font, 1.3, (221, 221, 221), 1, cv2.LINE_AA)

    else:
        cv2.putText(rendered_frame, 'Week of the {}'.format(date), (40, 150), font, 1.3, (221, 221, 221), 1, cv2.LINE_AA)

    # displays the legend of the colour code
    cv2.putText(rendered_frame, 'Origin and destination', (35, 260), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(rendered_frame, (40, 290), 10, (141, 91, 67), -1)
    cv2.putText(rendered_frame, 'Identical', (60, 300), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(rendered_frame, (40, 320), 10, (135, 162, 34), -1)
    cv2.putText(rendered_frame, 'Distinct', (60, 330), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # displays the legend of the size of the circles
    cv2.putText(rendered_frame, 'Number of passengers', (35, 380), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    max_weight = compute_weight(map_type, max_pass, max_pass)
    cv2.circle(rendered_frame, (40, 420), max_weight, (255, 255, 255), 1)
    cv2.putText(rendered_frame, '{} passengers'.format(max_pass), (80, 420), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    min_weight = compute_weight(map_type, min_pass, max_pass)
    cv2.circle(rendered_frame, (40, 460), min_weight, (255, 255, 255), 1)
    cv2.putText(rendered_frame, '{} passenger'.format(min_pass), (80, 460), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def display_general_information_text(image, map_type, video_title):

    # note that these position are based on an image size of [1920, 1080]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # displays the name of the boroughs of the city
    if map_type == 'total':
        # name of borough Manhattan
        cv2.putText(image, 'Manhattan', (770, 360), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        # name of borough Brooklyn
        cv2.putText(image, 'Brooklyn', (1130, 945), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        # name of borough Staten Island
        cv2.putText(image, 'Staten Island', (595, 1030), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        # name of borough Queens
        cv2.putText(image, 'Queens', (1480, 590), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        # name of borough Bronx
        cv2.putText(image, 'Bronx', (1370, 195), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    else:
        video_title = video_title + ' in ' + map_type

    # displays the title of the video
    # cv2.putText(image, video_title, (500, 1050), font, 1, (255, 255, 255), 2, cv2.LINE_AA)


def render_frame(frame, base_map, query_results, max_passenger, converted_shape_dict, map_type):

    # we make a copy of the map on which we will render the frame (each frame being
    # rendered on a new copy)
    map_rendered = base_map.copy()

    # we get each tuple from the query result, in the form (origin_id, dest_id, weight)
    for itinerary in query_results:
        zone_id_origin = convert_id_shape(itinerary[0])
        zone_id_destination = convert_id_shape(itinerary[1])

        weight = itinerary[2]
        weight = compute_weight(map_type, weight, max_passenger)

        # we get the coordinates of the center of the origin and the destination
        origin_coords = converted_shape_dict[zone_id_origin]['center']
        destination_coords = converted_shape_dict[zone_id_destination]['center']

        if frame == 0:
            # we start the rendering with the point at the origin
            # we convert to int as to be able to plot the point with opencv
            coords_point_to_draw = (int(origin_coords[0]), int(origin_coords[1]))

        else:
            # we extrapolate the position of the point between the origin and the
            # destination, as to have the point move from origin to destination
            # in 60 frames
            coords_point_to_draw = interpolate_next_position(origin_coords, destination_coords, 60, frame)

        x_point = coords_point_to_draw[0]
        y_point = coords_point_to_draw[1]

        if zone_id_origin == zone_id_destination:
            colour = (141, 91, 67)
        else:
            colour = (135, 162, 34)

        render_point_on_map(x_point, y_point, weight, map_rendered, colour)

    return map_rendered


def render_all_frames(render_frame_dict):

    # we extract the arguments we need from the input dictionary
    query_date = render_frame_dict['query_date']
    query_results = render_frame_dict['query_results']
    database = render_frame_dict['database']
    base_map = render_frame_dict['base_map']
    converted_shape_dict = render_frame_dict['converted_shape_dict']
    map_type = render_frame_dict['map_type']
    frames = render_frame_dict['frames']
    min_pass = render_frame_dict['min_passenger']
    max_pass = render_frame_dict['max_passenger']
    agg_per = render_frame_dict['agg_per']

    # we use the results of the query to render 60 frames
    # we want to render an animation of 1 second per given date, at 60 fps.
    for frame in range(0, 60):
        rendered_frame = render_frame(frame, base_map, query_results, max_pass, converted_shape_dict, map_type)

        # we display frame related text
        date_info = (agg_per, query_date)
        display_specific_text_animation(rendered_frame, date_info, map_type, min_pass, max_pass)

        frames.append(rendered_frame)

    return frames


def make_video_animation(frames, image_size, map_type):

    # Build the title for the animation
    if map_type == 'total':
        title = 'Animation_{}.avi'.format('NYC')
    else:
        title = 'Animation_{}.avi'.format(map_type)

    animation = cv2.VideoWriter(title, cv2.VideoWriter_fourcc(*'DIVX'), 30, image_size)
    # video title, codec, fps, frame size

    for i in range(len(frames)):
        animation.write(frames[i])

    animation.release()


def process_query_arg(render_animation_dict):
    period = render_animation_dict['period']
    query_dict = render_animation_dict['query_dict']
    database = render_animation_dict['database']
    specific_weekdays = query_dict['specific_weekdays']
    date = query_dict['date']
    aggregate_period = render_animation_dict['aggregate_period']
    weekdays = render_animation_dict['weekdays']

    query_results_dict = {}

    if aggregate_period is False and query_dict['date'] == 'loop_through_period':
        # in this case we want the result for each day of the period provided
        # if we have the flag loop_through_period in the query dict, it means the period
        # set for the query is multiple dates

        daterange = pd.date_range(period[0], period[1])
        # we run queries for each date in the daterange specified
        for single_date in daterange:
            date = pd.to_datetime(single_date)
            if specific_weekdays == 'on_specific_weekdays':

                # we check if the date of the daterange matches the weekday(s) we target
                if date.dayofweek in weekdays:
                    single_date = date.date().strftime('%Y-%m-%d')
                    query_dict['date'] = single_date
                    query = prepare_sql_query(query_dict)
                    query_results = make_sql_query(query, database)
                    query_results_dict[query_dict['date']] = query_results

                else:
                    # if a date in the range is not among the weekdays we want, we skip it
                    continue
            else:
                single_date = date.date().strftime('%Y-%m-%d')
                query_dict['date'] = single_date
                query = prepare_sql_query(query_dict)
                query_results = make_sql_query(query, database)
                query_results_dict[query_dict['date']] = query_results

    elif aggregate_period is True and query_dict['date'] == 'loop_through_period':
        # in this case, we want to aggregate the results (sum) per week
        daterange = pd.date_range(period[0], period[1])
        start_date = pd.to_datetime(period[0])
        end_date = pd.to_datetime(period[1])

        # let's build a list of all intervals we will want to aggregate the data for
        all_aggr_init = []
        start = start_date
        end = end_date

        # we add one list of dates per week to the list of all intervals
        i = 0
        for date in daterange:
            # we handle separately the first date of the period
            if i == 0:
                curr_week = [start.date().strftime('%Y-%m-%d')]

            if date != start_date and date != end_date:
                start_week_number = start.isocalendar()[1]
                date_week_number = date.isocalendar()[1]

                if date_week_number == start_week_number:
                    curr_week.append(date.date().strftime('%Y-%m-%d'))
                    i += 1
                else:
                    start = date
                    all_aggr_init.append(curr_week)
                    i = 0

        # we handle separately the last date of the period
        if curr_week not in all_aggr_init:
            curr_week.append(end_date.date().strftime('%Y-%m-%d'))
            all_aggr_init.append(curr_week)
        else:
            curr_week = [end_date.date().strftime('%Y-%m-%d')]
            all_aggr_init.append(curr_week)

        # now we keep only the first and last item of each interval

        all_aggr = []
        for interval in all_aggr_init:
            interval_new = [interval[0], interval[-1]]
            all_aggr.append(interval_new)

        # we now query for each interval
        for interval in all_aggr:
            query_dict['date'] = interval
            query = prepare_sql_query(query_dict)
            query_results = make_sql_query(query, database)
            query_results_dict[query_dict['date'][0]] = query_results

    else:
        # we have a single date to render for, so nothing to aggregate!
        # just in case we check that there is no mismatch between the single day and the
        # argument containing specific weekdays restrictions if any
        if specific_weekdays == 'on_specific_weekdays':

            # we check if the date of the daterange matches the weekday(s) we target
            date = pd.Timestamp(query_dict['date'])

            if date.dayofweek in weekdays:
                query = prepare_sql_query(query_dict)
                query_results = make_sql_query(query, database)
                query_results_dict[query_dict['date']] = query_results

            else:
                print("The date selected does not match the weekday(s) indicated. Please select either an interval ('time_granularity': 'period') or a valid weekday(s) list.")

        else:
            query = prepare_sql_query(query_dict)
            query_results = make_sql_query(query, database)
            query_results_dict[query_dict['date']] = query_results

    return query_results_dict


def render_animation_query_output(render_animation_dict):

    # We extract the variables we will need from the input dictionary
    query_dict = render_animation_dict['query_dict']
    query_results_dict = render_animation_dict['query_results_dict']
    base_map = render_animation_dict['base_map']
    map_type = render_animation_dict['map_type']
    shape_dict = render_animation_dict['shape_dict']
    df_sf = render_animation_dict['df_sf']
    database = render_animation_dict['database']
    image_size = render_animation_dict['image_size']
    render_single_borough = render_animation_dict['render_single_borough']
    min_passenger = render_animation_dict['min_passenger']
    max_passenger = render_animation_dict['max_passenger']
    aggregate_period = render_animation_dict['aggregate_period']

    if query_dict['filter_query_on_borough'] is False:
        # in this case, we may want the base map to be reduced to map_type, but the query
        # to be performed on the whole city - thus we want to represent points that may
        # not be inside the shape of the reduced base map
        projection = render_animation_dict['projection']
        converted_shape_dict = convert_shape_boundaries(shape_dict, projection)

    else:
        # we isolate the set of zones we want to draw points for in the right coordinate system
        converted_shape_dict, projection = get_shape_set_to_draw(map_type, shape_dict, df_sf, image_size)

    # we build a dictionary for the details of the rendering of each frame
    render_frame_dict = {'database': database, 'min_passenger': min_passenger,
                         'max_passenger': max_passenger, 'base_map': base_map,
                         'converted_shape_dict': converted_shape_dict,
                         'map_type': map_type, 'frames': [], 'agg_per': aggregate_period}

    # we render frames depending on the results of the query and the period inputted
    for query_date in query_results_dict:
        render_frame_dict['query_date'] = query_date
        render_frame_dict['query_results'] = query_results_dict[query_date]
        frames = render_all_frames(render_frame_dict)
        render_frame_dict['frames'] = frames

    if map_type == 'total':
        print('Rendering the results for NYC...')
    else:
        print('Rendering the results for {}...'.format(map_type))

    # we compile the video from all frames
    make_video_animation(frames, image_size, map_type)

    if map_type == 'total':
        print('The video for NYC has been rendered')
    else:
        print('The video for {} has been rendered'.format(map_type))


def make_flow_animation(animation_dict):
    # we extract the variables from the input dictionary
    shp_path = animation_dict['shp_path']
    image_size = animation_dict['image_size']
    map_to_render = animation_dict['map_to_render']
    render_single_borough = animation_dict['render_single_borough']
    title = animation_dict['title']
    database = animation_dict['db']
    data_table = animation_dict['data_table']
    lookup_table = animation_dict['lookup_table']
    aggregated_result = animation_dict['aggregated_result']
    filter_query_on_borough = animation_dict['filter_query_on_borough']
    time_granularity = animation_dict['time_granularity']
    period = animation_dict['period']
    weekdays = animation_dict['weekdays']
    aggregate_period = animation_dict['aggregate_period']

    # First import the shapefile and build the boundaries dictionary
    print('Building the base map...')
    shp_path = shp_path
    sf_nyc = shp.Reader(shp_path)
    df_sf = shp_to_df(sf_nyc)
    shape_boundaries = process_shape_boundaries(df_sf, sf_nyc)

    # optional fool_proof check
    # if filter on borough is not False, then it contains the name of a borouhgh,
    # that happens to be the only one we want to use to draw the base map
    # so we ignore the input of the user in the map_to_render argument
    if filter_query_on_borough is not False:
        map_to_render = [filter_query_on_borough]

    # Draw the base map and keep it in a saved variable
    base_maps = []
    if len(map_to_render) == 1:
        map_type = map_to_render[0]
        # we want to render on a single map
        draw_dict = {'image_size': image_size, 'render_single_borough': render_single_borough,
                     'map_type': map_type, 'title': title,
                     'shape_dict': shape_boundaries, 'df_sf': df_sf}
        base_map, projection = draw_base_map(draw_dict)
        base_maps.append((map_type, base_map, projection))

    else:
        # we want to render multiple animations at once, for different base maps
        for single_map in map_to_render:
            map_type = single_map
            draw_dict = {'image_size': image_size, 'render_single_borough': render_single_borough,
                         'map_type': map_type, 'title': title,
                         'shape_dict': shape_boundaries, 'df_sf': df_sf}
            base_map, projection = draw_base_map(draw_dict)
            base_maps.append((map_type, base_map, projection))

    # we define the render_animation_dict
    render_animation_dict = {'time_granularity': time_granularity, 'period': period, 'weekdays': weekdays,
                             'filter_query_on_borough': filter_query_on_borough, 'image_size': image_size,
                             'shape_dict': shape_boundaries, 'df_sf': df_sf, 'database': database,
                             'data_table': data_table, 'lookup_table': lookup_table,
                             'aggregated_result': aggregated_result,
                             'render_single_borough': render_single_borough,
                             'video_title': title, 'aggregate_period': aggregate_period}

    # we query the database
    print('Querying the dabase...')

    query_dict = build_query_dict(render_animation_dict)
    render_animation_dict['query_dict'] = query_dict
    query_results_dict = process_query_arg(render_animation_dict)

    # we find the min and max passengers for the whole year
    min_passenger = 999999999
    max_passenger = 0
    for query_date in query_results_dict:
        temp_min, temp_max = compute_min_max_passengers(query_results_dict[query_date], 2)
        if temp_min < min_passenger:
            min_passenger = temp_min
        if temp_max > max_passenger:
            max_passenger = temp_max

    render_animation_dict['query_results_dict'] = query_results_dict
    render_animation_dict['min_passenger'] = min_passenger
    render_animation_dict['max_passenger'] = max_passenger

    # we render the animation!
    for map_type, base_map, projection in base_maps:
        # we add variabled to the render frame dictionary
        render_animation_dict['base_map'] = base_map
        render_animation_dict['projection'] = projection
        render_animation_dict['map_type'] = map_type

        render_animation_query_output(render_animation_dict)


def prepare_heat_map_sql_query(query_dict):

    # We extract the variables we will need from the input dictionary
    data_table = query_dict['data_table']
    lookup_table = query_dict['lookup_table']
    aggregated_result = query_dict['aggregated_result']
    date = query_dict['date']
    filter_query_on_borough = query_dict['filter_query_on_borough']
    weekdays_vs_weekends = query_dict['specific_weekdays']

    # first we synthesise what we want to fetch
    if aggregated_result == 'count':
        # we will want to return the sum of count on the period
        aggregated_result = 'COUNT(passenger_count_per_day)'
    elif aggregated_result == 'avg':
        # we will want to return the average of count on the period
        aggregated_result = 'AVG(passenger_count_per_day)'

    # we prepare the period statements
    if type(date) == str:
        # in this case, we want the result on a single day
        date_statement = ("pickup_date = '{}'").format(date)
    else:
        # we provided a time interval we want the average of the aggregated_result on the
        # period
        start_date = date[0]
        end_date = date[1]
        date_statement = ("pickup_date BETWEEN '{}' AND '{}'").format(start_date, end_date)

    # we build the query
    if weekdays_vs_weekends == 'weekdays_vs_weekends':
        # in this situation we want to query 'separately' the values in weekdays and weekends
        # and make a difference on the average of the aggregated_result on the period
        date_statement_weekdays = ("pickup_date BETWEEN '{}' AND '{}' AND pickup_weekday IN (0, 1, 2, 3, 4)".format(start_date, end_date))
        date_statement_weekends = ("pickup_date BETWEEN '{}' AND '{}' AND pickup_weekday IN (5, 6)".format(start_date, end_date))

        # Case 1: we want to compare weekdays and weekends flow for a specific borough
        if filter_query_on_borough is not False:
            query = ("SELECT pu_id, do_id, diff \
                    FROM (SELECT wd_pu_id pu_id, wd_do_id do_id, wd_aggregated_result - we_aggregated_result diff\
                        FROM(SELECT CASE WHEN wd_pu_id IS NULL THEN we_pu_id ELSE wd_pu_id END AS wd_pu_id, \
                                    CASE WHEN wd_do_id IS NULL THEN we_do_id ELSE wd_do_id END AS wd_do_id,\
                                    CASE WHEN wd_aggregated_result IS NULL THEN 0 ELSE wd_aggregated_result END AS wd_aggregated_result,\
                                    CASE WHEN we_pu_id IS NULL THEN wd_pu_id ELSE we_pu_id END AS we_pu_id, \
                                    CASE WHEN we_do_id IS NULL THEN wd_do_id ELSE we_do_id END AS we_do_id,\
                                    CASE WHEN we_aggregated_result IS NULL THEN 0 ELSE we_aggregated_result END AS we_aggregated_result\
                        FROM (SELECT *\
                                FROM (SELECT PULocationID wd_pu_id, DOLocationID wd_do_id, {0} wd_aggregated_result\
                                    FROM {1}\
                                    WHERE {2} \
                                    GROUP BY wd_pu_id, wd_do_id) as weekdays\
                                LEFT JOIN (SELECT PULocationID we_pu_id, DOLocationID we_do_id, {0} we_aggregated_result\
                                        FROM {1}\
                                        WHERE {3} \
                                        GROUP BY we_pu_id, we_do_id) as weekends\
                                ON weekdays.wd_pu_id = weekends.we_pu_id \
                                    AND weekdays.wd_do_id = weekends.we_do_id\
                            UNION \
                                SELECT *\
                                FROM (SELECT PULocationID wd_pu_id, DOLocationID wd_do_id, {0} wd_aggregated_result\
                                        FROM {1}\
                                        WHERE {2} \
                                        GROUP BY wd_pu_id, wd_do_id) as weekdays\
                                RIGHT JOIN (SELECT PULocationID we_pu_id, DOLocationID we_do_id, {0} we_aggregated_result\
                                            FROM {1}\
                                            WHERE {3} \
                                            GROUP BY we_pu_id, we_do_id) as weekends\
                                ON weekdays.wd_pu_id = weekends.we_pu_id \
                                 AND weekdays.wd_do_id = weekends.we_do_id) as tab_1) as tab_2\
                    JOIN {4} lookup_pu\
                    ON lookup_pu.LocationID = tab_2.pu_id \
                    JOIN {4} lookup_do \
                    ON lookup_do.LocationID = tab_2.do_id \
                    WHERE lookup_pu.borough_name = '{5}' AND lookup_do.borough_name = '{5}'\
                    GROUP BY pu_id, do_id, diff;".format(aggregated_result, data_table,
                                                         date_statement_weekdays,
                                                         date_statement_weekends,
                                                         lookup_table,
                                                         filter_query_on_borough))

        # Case 2: we want to compare weekdays and weekends flow for the whole city
        else:
            query = ("SELECT wd_pu_id pu_id, wd_do_id do_id, wd_aggregated_result - we_aggregated_result diff\
                    FROM(SELECT CASE WHEN wd_pu_id IS NULL THEN we_pu_id ELSE wd_pu_id END AS wd_pu_id, \
                                CASE WHEN wd_do_id IS NULL THEN we_do_id ELSE wd_do_id END AS wd_do_id,\
                                CASE WHEN wd_aggregated_result IS NULL THEN 0 ELSE wd_aggregated_result END AS wd_aggregated_result,\
                                CASE WHEN we_pu_id IS NULL THEN wd_pu_id ELSE we_pu_id END AS we_pu_id, \
                                CASE WHEN we_do_id IS NULL THEN wd_do_id ELSE we_do_id END AS we_do_id,\
                                CASE WHEN we_aggregated_result IS NULL THEN 0 ELSE we_aggregated_result END AS we_aggregated_result\
                    FROM (SELECT *\
                            FROM (SELECT PULocationID wd_pu_id, DOLocationID wd_do_id, {0} wd_aggregated_result\
                                FROM {1}\
                                WHERE {2} \
                                GROUP BY wd_pu_id, wd_do_id) as weekdays\
                            LEFT JOIN (SELECT PULocationID we_pu_id, DOLocationID we_do_id, {0} we_aggregated_result\
                                    FROM {1}\
                                    WHERE {3} \
                                    GROUP BY we_pu_id, we_do_id) as weekends\
                            ON weekdays.wd_pu_id = weekends.we_pu_id \
                                AND weekdays.wd_do_id = weekends.we_do_id\
                        UNION \
                            SELECT *\
                            FROM (SELECT PULocationID wd_pu_id, DOLocationID wd_do_id, {0} wd_aggregated_result\
                                    FROM {1}\
                                    WHERE {2} \
                                    GROUP BY wd_pu_id, wd_do_id) as weekdays\
                            RIGHT JOIN (SELECT PULocationID we_pu_id, DOLocationID we_do_id, {0} we_aggregated_result\
                                        FROM {1}\
                                        WHERE {3} \
                                        GROUP BY we_pu_id, we_do_id) as weekends\
                            ON weekdays.wd_pu_id = weekends.we_pu_id \
                             AND weekdays.wd_do_id = weekends.we_do_id) as tab_1) as tab_2;".format(aggregated_result, data_table, date_statement_weekdays, date_statement_weekends))

    else:
        # Case 3: we want the total average/count on the period for a specific borough
        if filter_query_on_borough is not False:
            query = ("SELECT pu_id, do_id, {0} aggregated_result \
                    FROM \
                         (SELECT PULocationID pu_id, DOLocationID do_id, \
                                 passenger_count_per_day\
                        FROM {1}\
                        WHERE {2} \
                        GROUP BY pu_id, do_id) as tab_1 \
                    JOIN {3} lookup_pu\
                    ON lookup_pu.LocationID = tab_1.pu_id \
                    JOIN {3} lookup_do \
                    ON lookup_do.LocationID = tab_1.do_id \
                    WHERE lookup_pu.borough_name = '{4}' AND lookup_do.borough_name = '{4}'\
                    GROUP BY pu_id, do_id".format(aggregated_result, data_table, date_statement, lookup_table, filter_query_on_borough))

        # Case 4: we want the total average/count on the period for the whole city
        else:
            query = ("SELECT PULocationID pu_id, DOLocationID do_id, {0} aggregated_result\
                    FROM {1}\
                    WHERE {2} \
                    GROUP BY pu_id, do_id".format(aggregated_result, data_table, date_statement))

    return query


def process_heat_map_query_results(query_results):

    incoming_flow = {}
    outgoing_flow = {}

    # then we build a dictionary of outgoing traffic i.e each zone_id used
    # as a key in the dict will have a count of people going to another zone
    for itinerary in query_results:
        origin_id = itinerary[0]
        destination_id = itinerary[1]
        weight = itinerary[2]
        if origin_id not in outgoing_flow:
            outgoing_flow[origin_id] = []
        outgoing_flow[origin_id].append((destination_id, weight))

    # we finally do the same but with the incoming trafic i.e each zone_id used
    # as a key in the dict will have a count of people coming from another zone
    for itinerary in query_results:
        origin_id = itinerary[0]
        destination_id = itinerary[1]
        weight = itinerary[2]
        if destination_id not in incoming_flow:
            incoming_flow[destination_id] = []
        incoming_flow[destination_id].append((origin_id, weight))

    return outgoing_flow, incoming_flow


def find_names(zone_id, df_sf):

    zone_name = df_sf[df_sf.index == zone_id]['zone'].item()
    borough_name = df_sf[df_sf.index == zone_id]['borough'].item()

    return zone_name, borough_name


def compute_color(weight, min_passenger, max_passenger):

    # we use a color palette that spans between two very different colors, the idea
    # being to be able to distinguish positive from negative values

    max_pos_colour = (100, 100, 255)  # shade of red
    min_pos_colour = (40, 40, 100)  # shade of red
    min_neg_colour = (0, 0, 0)  # shade of blue
    max_neg_colour = (210, 150, 90)  # shade of blue

    if weight == 0:
        color = [40, 40, 40]  # grey

    else:

        if min_passenger == max_passenger:
            # in this case we have basically one color to represent only
            if weight > 0:
                color = max_pos_colour

            else:
                color = max_neg_colour

        elif min_passenger >= 0 and max_passenger > 0:
            # in this case we draw everything in shades of red
            weight_norm = weight/max_passenger
            blue_index = (max_pos_colour[0]-min_pos_colour[0])*weight_norm + min_pos_colour[0]
            green_index = (max_pos_colour[1]-min_pos_colour[1])*weight_norm + min_pos_colour[1]
            red_index = (max_pos_colour[2]-min_pos_colour[2])*weight_norm + min_pos_colour[2]
            color = (blue_index, green_index, red_index)

        elif min_passenger < 0 and max_passenger <= 0:
            # in this case we draw everything in shades of blue
            weight_norm = weight/min_passenger
            blue_index = (max_neg_colour[0]-min_neg_colour[0])*weight_norm + min_neg_colour[0]
            green_index = (max_neg_colour[1]-min_neg_colour[1])*weight_norm + min_neg_colour[1]
            red_index = (max_neg_colour[2]-min_neg_colour[2])*weight_norm + min_neg_colour[2]
            color = (blue_index, green_index, red_index)

        else:
            # in this case the color depends on the sign of the weight
            # we call this function recursively
            if weight > 0:
                color = compute_color(weight, 0, max_passenger)

            else:
                color = compute_color(weight, min_passenger, 0)

    return color


def render_map(render_map_dict):

    # first we extract the arguments we are going to need
    map_to_render = render_map_dict['map_to_render']
    zone_id = render_map_dict['zone_id']
    trips_list = render_map_dict['trips_list']
    draw_dict = render_map_dict['draw_dict']
    shape_dict = draw_dict['shape_dict']
    draw_dict['map_type'] = map_to_render
    min_passenger = render_map_dict['min_passenger']
    max_passenger = render_map_dict['max_passenger']

    base_map, projection = draw_base_map(draw_dict)

    # we obtain the converted_shape_dict we want to use to draw the heat map
    converted_shape_dict = convert_shape_boundaries(shape_dict, projection)

    # we keep track of how many colors we use to plot the legend afterwards
    colors = []
    for linked_zone in trips_list:
        id_shape_to_color = linked_zone[0]
        if id_shape_to_color != zone_id:
            weight = linked_zone[1]
            linked_shape = converted_shape_dict[id_shape_to_color]
            linked_points = linked_shape['points']
            pts = np.array(linked_points, np.int32)
            linked_color = compute_color(weight, min_passenger, max_passenger)
            if linked_color not in colors:
                colors.append(linked_color)
            cv2.fillPoly(base_map, [pts], linked_color)
            cv2.polylines(base_map, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)

    # we highlight the focused shape
    target_shape = converted_shape_dict[zone_id]
    target_points = target_shape['points']
    pts = np.array(target_points, np.int32)
    target_color = [95, 240, 255]
    cv2.polylines(base_map, [pts], True, target_color, 3, cv2.LINE_AA)

    return base_map, colors


 def display_scale_legend(map_image, font, min_pass, max_pass, colors):
    # we dynamically print a legend using a fixed step between two colors plotted
    
    k = 0
    top_bar_x = 30
    top_bar_y = 440
    
    #we add a legend for no passengers traveling
    cv2.rectangle(map_image,(top_bar_x, top_bar_y),(top_bar_x+40, top_bar_y + 20),(255, 255, 255),1)
    cv2.putText(map_image, 'No flow of people', (top_bar_x + 70, top_bar_y + 15), font, 0.7, (221, 221, 221), 1, cv2.LINE_AA)
    top_bar_y = top_bar_y + 22   
        
    #we prepare the ground to plot a dynamic legend for the colors
    if len(colors) < 8:
        scale_step = len(colors)
    else:
        scale_step = 8
    
    levels = []
    while k < scale_step:
        if scale_step > 1:
            level = max_pass + (min_pass - max_pass) * k/(scale_step-1)
        else:
            level = max_pass
        levels.append(level)
        k+=1
    
    #we check if there are negative and positive values to represent and if we already
    #have a 0 to represent ; if not, we will add it to the list of steps to plot
    neg_value_count = 0
    pos_value_count = 0
    zero_count = 0
    for level in levels:
        if level < 0:
            neg_value_count+= 1
        elif level == 0:
            zero_count+=1
        else:
            pos_value_count+=1
    
    if zero_count == 0:
        if neg_value_count > 0 and pos_value_count> 0:
            levels.append(0)
    
    #we plot dynamically the legend
    levels.sort()
    for level in levels:   
        color = compute_color(level, min_pass, max_pass)
        level = "{0:.2f}".format(level)
        cv2.rectangle(map_image,(top_bar_x, top_bar_y),(top_bar_x+40, top_bar_y + 20),color,-1)
        if float(level) == 0 or abs(float(level)) == 1:
            cv2.putText(map_image, '{} passenger'.format(level), (top_bar_x + 70, top_bar_y + 15), font, 0.7, (221, 221, 221), 1, cv2.LINE_AA)
        else:
            cv2.putText(map_image, '{} passengers'.format(level), (top_bar_x + 70, top_bar_y + 15), font, 0.7, (221, 221, 221), 1, cv2.LINE_AA)
        top_bar_y = top_bar_y + 20
    
      

def display_specific_text_heat_map(map_image, time_granularity, zone_info, min_pass, max_pass, colors):
    
    #note that these position are based on an image size of [1920, 1080]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if time_granularity == 'period':
        time_granularity_1 = 'Flow over the whole year'
        time_granularity_2 = ""
    else:
        time_granularity_1 = "Difference between weekdays"
        time_granularity_2 = "and weekends flow"
    
    #display the main title
    cv2.putText(map_image, time_granularity_1, (30, 150), font, 1.3, (221, 221, 221), 1, cv2.LINE_AA)
    cv2.putText(map_image, time_granularity_2, (30, 180), font, 1.3, (221, 221, 221), 1, cv2.LINE_AA)
    
    #display the zone id and name
    cv2.putText(map_image, '{} - '.format(zone_info[0]), (30, 240), font, 1.3, (221, 221, 221), 1, cv2.LINE_AA)
    cv2.putText(map_image, '{}'.format(zone_info[1]), (170, 240), font, 1.3, (221, 221, 221), 1, cv2.LINE_AA)
    
    #displays the legend of the colour code
    cv2.putText(map_image, 'Legend', (30,320), font, 0.9, (221, 221, 221), 1, cv2.LINE_AA)
    cv2.rectangle(map_image,(30,340),(70,360),(95, 240, 255),3)
    cv2.putText(map_image, 'Target zone', (100, 360), font, 0.7, (221, 221, 221), 1, cv2.LINE_AA)
    cv2.putText(map_image, 'Average number of passengers', (30, 410), font, 0.7, (221, 221, 221), 1, cv2.LINE_AA)
    cv2.putText(map_image, '* A negative value means more flow on weekends', (30, 430), font, 0.5, (221, 221, 221), 1, cv2.LINE_AA)
    
    display_scale_legend(map_image, font, min_pass, max_pass, colors)

    


def render_heat_map_query_output(render_heat_map_dict):
    
    draw_dict = render_heat_map_dict['draw_dict']
    flow_dict = render_heat_map_dict['flow_dict']
    flow_dir = render_heat_map_dict['flow_dir']
    time_granularity = render_heat_map_dict['time_granularity']
    df_sf = draw_dict['df_sf']
    
    
    for zone_id in flow_dict:
        #we ensure the ids are in the write 'system'
        trips_list = flow_dict[zone_id]
        i = 0
        for trip in trips_list:
            dest_id = convert_id_shape(trip[0])
            trips_list[i] = (dest_id, trip[1])
            i+=1
        zone_id = convert_id_shape(zone_id)
        
        
        #first let's figure out in which borough it is, and which name it has
        zone_name, borough_name = find_names(zone_id, df_sf)
        
        #we want to render the base map on the whole NYC map, as well as on a borough
        #zoomed map
        
        #Let's build the file names
        zone_id_lookup = convert_id_shape(zone_id, inverse=True)
        if time_granularity == 'weekdays_vs_weekends':
            nyc_file_name = 'NYC_{}_{}_{}_2018_diff_WD_WE'.format(zone_id_lookup, zone_name, flow_dir)
            borough_file_name = '{}_{}_{}_{}_2018_diff_WD_WE'.format(borough_name, zone_id_lookup, 
                                                                     zone_name, flow_dir)
        else:
            nyc_file_name = 'NYC_{}_{}_{}_2018'.format(zone_id_lookup, zone_name, flow_dir)
            borough_file_name = '{}_{}_{}_{}_2018'.format(borough_name,zone_id_lookup, 
                                                          zone_name, flow_dir)
        
        zone_info = [zone_id_lookup, zone_name]
        
        #we get the min and max number of passengers and color the linked zones
        min_passenger, max_passenger = compute_min_max_passengers(trips_list, 1)
    
        #Render results on the NYC map
        render_map_dict_NYC = {'map_to_render':'total', 'zone_id': zone_id, 
                               'draw_dict':draw_dict, 'min_passenger':min_passenger, 
                               'max_passenger':max_passenger, 'trips_list':trips_list}
        
        nyc_map, nyc_colors = render_map(render_map_dict_NYC)
        
        #display the legend
        display_specific_text_heat_map(nyc_map, time_granularity, zone_info, 
                                       min_passenger, max_passenger, nyc_colors)

        #save the image
        cv2.imwrite(('{}.png').format(nyc_file_name),nyc_map)
        
    

        #Render results on the borough map
        render_map_dict_borough = {'map_to_render':borough_name, 'zone_id': zone_id, 
                                   'draw_dict':draw_dict, 'min_passenger':min_passenger, 
                                   'max_passenger':max_passenger, 'trips_list':trips_list}
        
        borough_map, borough_colors = render_map(render_map_dict_borough)
        
        #display the legend
        display_specific_text_heat_map(borough_map, time_granularity, zone_info, 
                                       min_passenger, max_passenger, borough_colors)
        

        #save the image
        cv2.imwrite(('{}.png').format(borough_file_name),borough_map)
        


def make_heat_map(heat_map_dict):
    
    #we extract the variables from the input dictionary
    shp_path = heat_map_dict['shp_path']
    image_size = heat_map_dict['image_size']
    render_single_borough = heat_map_dict['render_single_borough']
    title = heat_map_dict['title']
    database = heat_map_dict['db']
    data_table = heat_map_dict['data_table']
    lookup_table = heat_map_dict['lookup_table']
    aggregated_result = heat_map_dict['aggregated_result']
    filter_query_on_borough = heat_map_dict['filter_query_on_borough']
    period = heat_map_dict['period']
    
    if heat_map_dict['weekdays_vs_weekends']==True:
        time_granularity = 'weekdays_vs_weekends'
    else:
        time_granularity = 'period'
    
    
    #Import the shapefile and build the boundaries dictionary
    shp_path = shp_path
    sf_nyc = shp.Reader(shp_path)
    df_sf = shp_to_df(sf_nyc)
    shape_boundaries = process_shape_boundaries(df_sf, sf_nyc)
    
    #we define the render_heat_map_dict    
    render_heat_map_dict = {'time_granularity':time_granularity, 'period':period,  
                             'image_size':image_size,'data_table':data_table, 
                             'lookup_table':lookup_table,'aggregated_result':aggregated_result,
                             'title':title, 'filter_query_on_borough':filter_query_on_borough}
    
    #we build the query statement and execute it on the database
    print('Querying the dabase...')
    query_dict = build_query_dict(render_heat_map_dict)
    
    if query_dict['date'] == 'loop_through_period':
        #if we have the flag loop_through_period in the query dict, it means the period
        #set for the query is multiple dates, therefore we want the query to return an
        #average on a time interval, and not on a single date
        period = render_heat_map_dict['period']
        daterange = pd.date_range(period[0],period[1])
        query_dict['date'] = period
    
    query = prepare_heat_map_sql_query(query_dict)
    query_results = make_sql_query(query, database)
    
    
    #we process the query results
    outgoing_flow, incoming_flow = process_heat_map_query_results(query_results)

    draw_dict = {'image_size':image_size, 'render_single_borough':render_single_borough, 
             'title':title, 'shape_dict':shape_boundaries, 'df_sf':df_sf}
    
    print('Building the outgoing maps...')
    #we build the maps for the outgoing flow
    render_heat_map_dict_out = {'draw_dict':draw_dict, 'flow_dict':outgoing_flow, 
                            'flow_dir': 'out','time_granularity':time_granularity}
    
    render_heat_map_query_output(render_heat_map_dict_out)  
    
    print('Building the incoming maps...')
    #we build the maps for the incoming flow
    render_heat_map_dict_in = {'draw_dict':draw_dict, 'flow_dict':incoming_flow, 
                            'flow_dir': 'in','time_granularity':time_granularity}
    
    render_heat_map_query_output(render_heat_map_dict_in) 
  






shp_path = "/Users/acoullandreau/Desktop/Taxi_rides_DS/taxi_zones/taxi_zones.shp"

animation_dict_2018 = {'shp_path':shp_path, 'image_size':(1920,1080), 
                       'map_to_render':['total', 'Manhattan', 'Bronx', 'Queens', 'Staten Island', 'Brooklyn'],
                       'render_single_borough':False,
                       'filter_query_on_borough':False,
                       'title':'General flow of passengers in 2018', 
                       'db':'nyc_taxi_rides', 'data_table':'taxi_rides_2018',
                       'lookup_table':'taxi_zone_lookup_table', 'aggregated_result':'count',
                       'time_granularity':'period', 'period':['2018-01-01','2018-12-31'],  
                       'weekdays':(), 'aggregate_period':False}



shp_path = "/Users/acoullandreau/Desktop/Taxi_rides_DS/taxi_zones/taxi_zones.shp"

heat_map_dict = {'shp_path':shp_path, 'image_size':(1920,1080),'db':'nyc_taxi_rides', 
                 'data_table':'passenger_count_2018','lookup_table':'taxi_zone_lookup_table', 
                 'aggregated_result':'count', 'weekdays_vs_weekends':False,
                 'period':['2018-01-01','2018-01-31'], 'render_single_borough':False,
                  'filter_query_on_borough':False, 'title':'Chloropeth map over the year'} 

make_heat_map(heat_map_dict)




