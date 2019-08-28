import cv2


class PointOnMap:

    def __init__(self, coordinates, weight, color):
        self.x_coord_or = coordinates[0]
        self.y_coord_or = coordinates[1]
        self.x_coord_curr = coordinates[0]
        self.y_coord_curr = coordinates[1]
        self.weight = weight
        self.color = color

    def render_point_on_map(self, base_map):
        """
            Renders the point on the map file provided as an argument
        """
        x = int(self.x_coord_curr)
        y = int(self.y_coord_curr)
        cv2.circle(base_map, (x, y), self.weight, self.color, -1)

    def interpolate_next_position(self, target_coords, tot_frames, curr_frame):
        """
            Interpolates the position of a point, knowing where we want it to
            arrive, in how many 'hops'
            Especially used when rendering multiple frames for an animation
        """
        # as to perform the arithmetic operations, we convert everything to
        # float for more precision
        x_origin = float(self.x_coord_or)
        y_origin = float(self.y_coord_or)
        x_destination = float(target_coords[0])
        y_destination = float(target_coords[1])
        tot_frames = float(tot_frames - 1)
        curr_frame = float(curr_frame)

        delta_x = (x_destination - x_origin)/tot_frames
        delta_y = (y_destination - y_origin)/tot_frames

        # the rendering with OpenCV demands integers values for the positioning
        # so we convert x and y to int
        self.x_coord_curr = int(x_origin+delta_x*curr_frame)
        self.y_coord_curr = int(y_origin+delta_y*curr_frame)