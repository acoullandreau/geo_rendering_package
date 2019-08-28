
class Projection:

    def __init__(self, map_to_scale, margin=[0, 0, 0, 0]):
        self.image_size = map_to_scale.image_size
        self.map_max_bound = map_to_scale.max_bound
        self.map_min_bound = map_to_scale.min_bound
        self.margin = margin  # (top, right, bottom, left) in pixels
        self.conversion = self.define_projection()[0]
        self.axis_to_center = self.define_projection()[1]

    def define_projection(self):
        """
            Calculates the conversion rate and axis on which to center the 
            converted coordinates
        """
        
        # We get the max 'coordinates' for both the target image and
        # the shape we want to draw
        image_x_max = self.image_size[0] - self.margin[1]
        image_y_max = self.image_size[1] - self.margin[2]
        image_x_min = 0 + self.margin[3]
        image_y_min = 0 + self.margin[0]
        map_x_max = self.map_max_bound[0]
        map_y_max = self.map_max_bound[1]
        map_x_min = self.map_min_bound[0]
        map_y_min = self.map_min_bound[1]

        # we check which size is bigger to know based on which axis we want
        # to scale our shape to
        # we do the comparison using the aspect ratio expectations (dividing
        # each axis by the size of the target axis in the new scale)
        ratio_x = (map_x_max - map_x_min)/(image_x_max - image_x_min)
        ratio_y = (map_y_max - map_y_min)/(image_y_max - image_y_min)
        if ratio_x > ratio_y:
            conversion = 1 / ratio_x
            axis_to_center = 'y'  # we store the axis we will want to center on
            # based on which axis we perform the scaling from
        else:
            conversion = 1 / ratio_y
            axis_to_center = 'x'

        return conversion, axis_to_center

    def apply_projection(self, coords, inverse=False):
        """
            applies the conversion on coordinates ;
            the inverse argument allows to go from one coordinate system
            (the original one), to the new one
        """
        x = coords[0]
        y = coords[1]
        x_min = self.map_min_bound[0]
        y_min = self.map_min_bound[1]

        if inverse is False:
            # to be able to center the image, we first translate the
            # coordinates to the origin
            x = (x - x_min) * self.conversion
            y = (y - y_min) * self.conversion
        else:
            x = x / self.conversion + x_min
            y = y / self.conversion + y_min

        coords = [x, y]
        return coords

    def apply_translation(self, coords):
        """
            Translates the coordinates along the axis to center in order to center the map
        """
        axis_to_center = self.axis_to_center
        image_x_max = self.image_size[0] - self.margin[1]
        image_y_max = self.image_size[1] - self.margin[2]
        image_x_min = 0 + self.margin[3]
        image_y_min = 0 + self.margin[0]
        map_x_max = self.map_max_bound[0]
        map_y_max = self.map_max_bound[1]
        map_max_converted = self.apply_projection((map_x_max, map_y_max))
        map_x_min = self.map_min_bound[0]
        map_y_min = self.map_min_bound[1]
        map_min_converted = self.apply_projection((map_x_min, map_y_min))

        if axis_to_center == 'x':
            map_x_max_conv = map_max_converted[0]
            map_x_min_conv = map_min_converted[0]
            center_translation = ((image_x_max - image_x_min)
                                  - (map_x_max_conv - map_x_min_conv))/2
        else:
            map_y_max_conv = map_max_converted[1]
            map_y_min_conv = map_min_converted[1]
            center_translation = ((image_y_max - image_y_min)
                                  - (map_y_max_conv - map_y_min_conv))/2

        # we center the map on the axis that was not used to scale the image
        if axis_to_center == 'x':
            coords[0] = coords[0] + center_translation
        else:
            coords[1] = coords[1] + center_translation

        # we mirror the image to match the axis alignment
        coords[1] = image_y_max - coords[1]

        return coords