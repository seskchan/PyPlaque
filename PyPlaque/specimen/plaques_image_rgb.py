import numpy as np

from ..specimen import PlaquesMask
from ..utils import fixed_threshold

class PlaquesImageRGB(PlaquesMask):
  """
  **PlaquesImageRGB Class** 
  The class is designed to hold RGB image data containing multiple plaque phenotypes with a 
  respective binary mask. The class inherits from PlaquesMask.
    
  Attributes:
    name (str, required): A string representing the name or identifier for the image sample. 
    
    image (np.ndarray, required): A 3D numpy array containing RGB image data of a virological 
                                plaque object, corresponding to the mask. 

    plaques_mask (np.ndarray, required): A 2D numpy array representing the binary mask of all 
                                      virological plaque objects. 

    use_picks (bool, optional): Indicates whether to use pick-based area calculation. 
                              Defaults to False.

  Raises:
    TypeError: If `name` is not a string, if `image` is not a 3D numpy array, or if `plaques_mask` 
    is not a 2D numpy array.
  """
  def __init__(self, 
               name: str, 
               image: np.ndarray, 
               plaques_mask: np.ndarray | None = None, 
               threshold = None,
               sigma = 5,
               use_picks:bool=False):
    # check types
    if not isinstance(name, str):
      raise TypeError("Image name atribute must be a str")
    if (not isinstance(image, np.ndarray)) or (not image.ndim == 3):
      raise TypeError("Image atribute must be a 3D (RGB) numpy array")
    if plaques_mask:
      if (not isinstance(plaques_mask, np.ndarray)) or (not plaques_mask.ndim
      == 2):
        raise TypeError("Mask atribute must be a 2D numpy array")
      self.plaques_mask = plaques_mask
    elif threshold and sigma:
      plaques_mask = fixed_threshold(image, threshold, sigma) # mask:RGB(x,y,3) #gaussian()/normalisation implemented along each axis
      # Compression
      # Avg Pooling

      self.plaques_mask = plaques_mask
    else:
      raise ValueError("Either mask or fixed threshold must be provided")

    # inherit super class (plaques_mask) __init__ with child name, plaques_mask and use_picks
    super(PlaquesImageRGB, self).__init__(name, plaques_mask,use_picks)
    #super().__init__(name, plaques_mask, use_picks) in python3
    self.image = image

if __name__ == __name__:
  test = np.random.rand(256,256,3)
  test_obj = PlaquesImageRGB(name="test",image=test, sigma=1)
  print(f"original: {test}")
  print(f"new: {test_obj.plaques_mask}")