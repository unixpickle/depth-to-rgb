from abc import ABC, abstractmethod


class Transcoder(ABC):
    """
    An object which converts between 16-bit depth images
    and 8-bit RGB images.
    """
    @abstractmethod
    def to_rgb(self, depth_image):
        """
        Convert a depth image to an RGB image.

        Args:
          depth_image: a 2-D uint16 numpy array.

        Returns:
          A 3-D uint8 numpy array.
        """
        pass

    @abstractmethod
    def to_depth(self, rgb_image):
        """
        Convert an RGB image to a depth image.

        Args:
          rgb_image: a 3-D uint8 numpy array.

        Returns:
          A 2-D uint16 numpy array.
        """
        pass
