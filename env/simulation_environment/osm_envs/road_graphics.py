from typing import TYPE_CHECKING, List, Tuple, Union

import numpy as np
import pygame

from simulation_environment.road.lane import AbstractLane, LineType, PolyLane
from simulation_environment.road.road import Road
from simulation_environment.osm_envs.vehicle_graphics import VehicleGraphics
from simulation_environment.vehicle.objects import Landmark, Obstacle

if TYPE_CHECKING:
    from simulation_environment.vehicle.objects import RoadObject

PositionType = Union[Tuple[float, float], np.ndarray]


class WorldSurface(pygame.Surface):

    """A pygame Surface implementing a local coordinate system so that we can move and zoom in the displayed area."""

    BLACK = (0, 0, 0)
    GREY = (100, 100, 100)
    GREEN = (50, 200, 0)
    YELLOW = (200, 200, 0)
    WHITE = (255, 255, 255)
    INITIAL_SCALING = 5.5
    INITIAL_CENTERING = [0.5, 0.5]
    SCALING_FACTOR = 1.3
    MOVING_FACTOR = 0.1

    def __init__(
        self, size: Tuple[int, int], flags: object, surf: pygame.SurfaceType
    ) -> None:
        super().__init__(size, flags, surf)
        self.origin = np.array([0, 0])
        self.scaling = self.INITIAL_SCALING
        self.centering_position = self.INITIAL_CENTERING

    def pix(self, length: float) -> int:
        """
        Convert a distance [m] to pixels [px].

        :param length: the input distance [m]
        :return: the corresponding size [px]
        """
        return int(length * self.scaling)

    def pos2pix(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert two world coordinates [m] into a position in the surface [px]

        :param x: x world coordinate [m]
        :param y: y world coordinate [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        return self.pix(x - self.origin[0]), self.pix(y - self.origin[1])

    def vec2pix(self, vec: PositionType) -> Tuple[int, int]:
        """
        Convert a world position [m] into a position in the surface [px].

        :param vec: a world position [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        return self.pos2pix(vec[0], vec[1])

    def is_visible(self, vec: PositionType, margin: int = 50) -> bool:
        """
        Is a position visible in the surface?
        :param vec: a position
        :param margin: margins around the frame to test for visibility
        :return: whether the position is visible
        """
        x, y = self.vec2pix(vec)
        return (
            -margin < x < self.get_width() + margin
            and -margin < y < self.get_height() + margin
        )

    def move_display_window_to(self, position: PositionType) -> None:
        """
        Set the origin of the displayed area to center on a given world position.

        :param position: a world position [m]
        """
        self.origin = position - np.array(
            [
                self.centering_position[0] * self.get_width() / self.scaling,
                self.centering_position[1] * self.get_height() / self.scaling,
            ]
        )

    def handle_event(self, event: pygame.event.EventType) -> None:
        """
        Handle pygame events for moving and zooming in the displayed area.

        :param event: a pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_l:
                self.scaling *= 1 / self.SCALING_FACTOR
            if event.key == pygame.K_o:
                self.scaling *= self.SCALING_FACTOR
            if event.key == pygame.K_m:
                self.centering_position[0] -= self.MOVING_FACTOR
            if event.key == pygame.K_k:
                self.centering_position[0] += self.MOVING_FACTOR


class LaneGraphics(object):

    """A visualization of a lane."""

    # See https://www.researchgate.net/figure/French-road-traffic-lane-description-and-specification_fig4_261170641
    STRIPE_SPACING: float = 4.33
    """ Offset between stripes [m]"""

    STRIPE_LENGTH: float = 3
    """ Length of a stripe [m]"""

    STRIPE_WIDTH: float = 0.3
    """ Width of a stripe [m]"""

    @classmethod
    def display(cls, lane: AbstractLane, surface: WorldSurface) -> None:
        """
        Display a lane on a surface.

        :param lane: the lane to be displayed
        :param surface: the pygame surface
        """
        stripes_count = int(
            2
            * (surface.get_height() + surface.get_width())
            / (cls.STRIPE_SPACING * surface.scaling)
        )
        s_origin, _ = lane.local_coordinates(surface.origin)
        s0 = (
            int(s_origin) // cls.STRIPE_SPACING - stripes_count // 2
        ) * cls.STRIPE_SPACING
        for side in range(2):
            if lane.line_types[side] == LineType.STRIPED:
                cls.striped_line(lane, surface, stripes_count, s0, side)
            elif lane.line_types[side] == LineType.CONTINUOUS:
                cls.continuous_curve(lane, surface, stripes_count, s0, side)
            elif lane.line_types[side] == LineType.CONTINUOUS_LINE:
                cls.continuous_line(lane, surface, stripes_count, s0, side)

    @classmethod
    def striped_line(
        cls,
        lane: AbstractLane,
        surface: WorldSurface,
        stripes_count: int,
        longitudinal: float,
        side: int,
    ) -> None:
        """
        Draw a striped line on one side of a lane, on a surface.

        :param lane: the lane
        :param surface: the pygame surface
        :param stripes_count: the number of stripes to draw
        :param longitudinal: the longitudinal position of the first stripe [m]
        :param side: which side of the road to draw [0:left, 1:right]
        """
        starts = longitudinal + np.arange(stripes_count) * cls.STRIPE_SPACING
        ends = (
            longitudinal
            + np.arange(stripes_count) * cls.STRIPE_SPACING
            + cls.STRIPE_LENGTH
        )
        lats = [(side - 0.5) * lane.width_at(s) for s in starts]
        cls.draw_stripes(lane, surface, starts, ends, lats)

    @classmethod
    def continuous_curve(
        cls,
        lane: AbstractLane,
        surface: WorldSurface,
        stripes_count: int,
        longitudinal: float,
        side: int,
    ) -> None:
        """
        Draw a striped line on one side of a lane, on a surface.

        :param lane: the lane
        :param surface: the pygame surface
        :param stripes_count: the number of stripes to draw
        :param longitudinal: the longitudinal position of the first stripe [m]
        :param side: which side of the road to draw [0:left, 1:right]
        """
        starts = longitudinal + np.arange(stripes_count) * cls.STRIPE_SPACING
        ends = (
            longitudinal
            + np.arange(stripes_count) * cls.STRIPE_SPACING
            + cls.STRIPE_SPACING
        )
        lats = [(side - 0.5) * lane.width_at(s) for s in starts]
        cls.draw_stripes(lane, surface, starts, ends, lats)

    @classmethod
    def continuous_line(
        cls,
        lane: AbstractLane,
        surface: WorldSurface,
        stripes_count: int,
        longitudinal: float,
        side: int,
    ) -> None:
        """
        Draw a continuous line on one side of a lane, on a surface.

        :param lane: the lane
        :param surface: the pygame surface
        :param stripes_count: the number of stripes that would be drawn if the line was striped
        :param longitudinal: the longitudinal position of the start of the line [m]
        :param side: which side of the road to draw [0:left, 1:right]
        """
        starts = [longitudinal + 0 * cls.STRIPE_SPACING]
        ends = [longitudinal + stripes_count * cls.STRIPE_SPACING + cls.STRIPE_LENGTH]
        lats = [(side - 0.5) * lane.width_at(s) for s in starts]
        cls.draw_stripes(lane, surface, starts, ends, lats)

    @classmethod
    def draw_stripes(
        cls,
        lane: AbstractLane,
        surface: WorldSurface,
        starts: List[float],
        ends: List[float],
        lats: List[float],
    ) -> None:
        """
        Draw a set of stripes along a lane.

        :param lane: the lane
        :param surface: the surface to draw on
        :param starts: a list of starting longitudinal positions for each stripe [m]
        :param ends: a list of ending longitudinal positions for each stripe [m]
        :param lats: a list of lateral positions for each stripe [m]
        """
        starts = np.clip(starts, 0, lane.length)
        ends = np.clip(ends, 0, lane.length)

        if isinstance(lane, PolyLane) and not getattr(lane, 'straight_line'):
            points_list = []
            last_idx = -1
            for k, _ in enumerate(starts):
                if abs(starts[k] - ends[k]) > 0.5 * cls.STRIPE_LENGTH:
                    points_list.append(surface.vec2pix(lane.position(starts[k], lats[k])))
                    last_idx = k
                    # points_list.append(surface.vec2pix(lane.position(ends[k], lats[k])))
            else:
                if last_idx > 0:
                    points_list.append(surface.vec2pix(lane.position(ends[k], lats[k])))
            pygame.draw.aalines(surface, surface.WHITE, False, points_list)
            # pygame.draw.aaline(surface, surface.WHITE,
            #                  (surface.vec2pix(lane.position(starts[k], lats[k]))),
            #                  (surface.vec2pix(lane.position(ends[k], lats[k]))),
            #                  max(surface.pix(cls.STRIPE_WIDTH), 1))
        else:
            color = surface.WHITE
            # if isinstance(lane, PolyLane) and getattr(lane, 'straight_line'):
            #     color = surface.BLUE
            for k, _ in enumerate(starts):
                if abs(starts[k] - ends[k]) > 0.5 * cls.STRIPE_LENGTH:
                    pygame.draw.aaline(surface, color,
                                       (surface.vec2pix(lane.position(starts[k], lats[k]))),
                                       (surface.vec2pix(lane.position(ends[k], lats[k]))),
                                       max(surface.pix(cls.STRIPE_WIDTH), 1))

    @classmethod
    def draw_ground(
        cls,
        lane: AbstractLane,
        surface: WorldSurface,
        color: Tuple[float],
        width: float,
        draw_surface: pygame.Surface = None,
    ) -> None:
        draw_surface = draw_surface or surface
        stripes_count = int(
            2
            * (surface.get_height() + surface.get_width())
            / (cls.STRIPE_SPACING * surface.scaling)
        )
        s_origin, _ = lane.local_coordinates(surface.origin)
        s0 = (
            int(s_origin) // cls.STRIPE_SPACING - stripes_count // 2
        ) * cls.STRIPE_SPACING
        dots = []
        for side in range(2):
            longis = np.clip(
                s0 + np.arange(stripes_count) * cls.STRIPE_SPACING, 0, lane.length
            )
            lats = [2 * (side - 0.5) * width for _ in longis]
            new_dots = [
                surface.vec2pix(lane.position(longi, lat))
                for longi, lat in zip(longis, lats)
            ]
            new_dots = reversed(new_dots) if side else new_dots
            dots.extend(new_dots)
        pygame.draw.polygon(draw_surface, color, dots, 0)


class RoadGraphics(object):

    """A visualization of a road lanes and vehicles."""

    @staticmethod
    def display(road: Road, surface: WorldSurface) -> None:
        """
        Display the road lanes on a surface.

        :param road: the road to be displayed
        :param surface: the pygame surface
        """
        surface.fill(surface.GREY)
        laneletmap = road.lanelet
        for k, v in laneletmap.items():

            type_dict = v['type']
            if type_dict['type'] == 'pedestrian_marking':
                continue

            ls = v['points']
            ls_points = [surface.vec2pix((pt[0], pt[1])) for pt in ls]
            if type_dict['color'] == 'black':
                type_dict['color'] = 'white'
            pygame.draw.aalines(surface, type_dict['color'].upper(), False, ls_points)

        for k, v in laneletmap.items():
            type_dict = v['type']
            if type_dict['type'] == 'pedestrian_marking':
                length = int(v['spline'].s[-1])
                w = 1
                h = 0.2
                for lth in np.linspace(1, length, length):
                    ls_points = []
                    for lat, lon in [(w, -h), (w, h), (-w, h), (-w, -h)]:
                        point = v['spline'].frenet_to_cartesian1D(lth + lon, lat)
                        ls_points.append(surface.vec2pix((point[0], point[1])))

                    pygame.draw.polygon(surface, type_dict['color'].upper(), ls_points, 0)

    @staticmethod
    def display_traffic(
        road: Road,
        surface: WorldSurface,
        simulation_frequency: int = 15,
        offscreen: bool = False,
    ) -> None:
        """
        Display the road vehicles on a surface.

        :param road: the road to be displayed
        :param surface: the pygame surface
        :param simulation_frequency: simulation frequency
        :param offscreen: render without displaying on a screen
        """
        if road.record_history:
            for v in road.vehicles:
                VehicleGraphics.display_history(
                    v, surface, simulation=simulation_frequency, offscreen=offscreen
                )
        for v in road.vehicles:
            VehicleGraphics.display(v, surface, offscreen=offscreen)

    @staticmethod
    def display_road_objects(
        road: Road, surface: WorldSurface, offscreen: bool = False
    ) -> None:
        """
        Display the road objects on a surface.

        :param road: the road to be displayed
        :param surface: the pygame surface
        :param offscreen: whether the rendering should be done offscreen or not
        """
        for o in road.objects:
            RoadObjectGraphics.display(o, surface, offscreen=offscreen)


class RoadObjectGraphics:

    """A visualization of objects on the road."""

    YELLOW = (200, 200, 0)
    BLUE = (100, 200, 255)
    RED = (255, 100, 100)
    GREEN = (50, 200, 0)
    BLACK = (60, 60, 60)
    DEFAULT_COLOR = YELLOW

    @classmethod
    def display(
        cls,
        object_: "RoadObject",
        surface: WorldSurface,
        transparent: bool = False,
        offscreen: bool = False,
    ):
        """
        Display a road objects on a pygame surface.

        The objects is represented as a colored rotated rectangle

        :param object_: the vehicle to be drawn
        :param surface: the surface to draw the object on
        :param transparent: whether the object should be drawn slightly transparent
        :param offscreen: whether the rendering should be done offscreen or not
        """
        o = object_
        s = pygame.Surface(
            (surface.pix(o.LENGTH), surface.pix(o.LENGTH)), pygame.SRCALPHA
        )  # per-pixel alpha
        rect = (
            0,
            surface.pix(o.LENGTH / 2 - o.WIDTH / 2),
            surface.pix(o.LENGTH),
            surface.pix(o.WIDTH),
        )
        pygame.draw.rect(s, cls.get_color(o, transparent), rect, 0)
        pygame.draw.rect(s, cls.BLACK, rect, 1)
        if (
            not offscreen
        ):  # convert_alpha throws errors in offscreen mode TODO() Explain why
            s = pygame.Surface.convert_alpha(s)
        h = o.heading if abs(o.heading) > 2 * np.pi / 180 else 0
        # Centered rotation
        position = surface.pos2pix(o.position[0], o.position[1])
        cls.blit_rotate(surface, s, position, np.rad2deg(-h))

    @staticmethod
    def blit_rotate(
        surf: pygame.SurfaceType,
        image: pygame.SurfaceType,
        pos,
        angle: float,
        origin_pos = None,
        show_rect: bool = False,
    ) -> None:
        """Many thanks to https://stackoverflow.com/a/54714144."""
        # calculate the axis aligned bounding box of the rotated image
        w, h = image.get_size()
        box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
        box_rotate = [p.rotate(angle) for p in box]
        min_box = (
            min(box_rotate, key=lambda p: p[0])[0],
            min(box_rotate, key=lambda p: p[1])[1],
        )
        max_box = (
            max(box_rotate, key=lambda p: p[0])[0],
            max(box_rotate, key=lambda p: p[1])[1],
        )

        # calculate the translation of the pivot
        if origin_pos is None:
            origin_pos = w / 2, h / 2
        pivot = pygame.math.Vector2(origin_pos[0], -origin_pos[1])
        pivot_rotate = pivot.rotate(angle)
        pivot_move = pivot_rotate - pivot

        # calculate the upper left origin of the rotated image
        origin = (
            pos[0] - origin_pos[0] + min_box[0] - pivot_move[0],
            pos[1] - origin_pos[1] - max_box[1] + pivot_move[1],
        )
        # get a rotated image
        rotated_image = pygame.transform.rotate(image, angle)
        # rotate and blit the image
        surf.blit(rotated_image, origin)
        # draw rectangle around the image
        if show_rect:
            pygame.draw.rect(surf, (255, 0, 0), (*origin, *rotated_image.get_size()), 2)

    @classmethod
    def get_color(cls, object_: "RoadObject", transparent: bool = False):
        color = cls.DEFAULT_COLOR

        if isinstance(object_, Obstacle):
            if object_.crashed:
                # indicates failure
                color = cls.RED
            else:
                color = cls.YELLOW
        elif isinstance(object_, Landmark):
            if object_.hit:
                # indicates success
                color = cls.GREEN
            else:
                color = cls.BLUE

        if transparent:
            color = (color[0], color[1], color[2], 30)

        return color
