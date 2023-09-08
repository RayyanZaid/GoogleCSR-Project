import numpy as np

class RectangularToPolar:

    def __init__(self, player_x : float, player_y : float, hoop_x : float, hoop_y: float):
        self.player_x : float = player_x
        self.player_y : float = player_y
        
        self.hoop_x = hoop_x
        self.hoop_y = hoop_y


    def returnMagnitudeR(self) -> float:

        hoop_point = np.array([self.hoop_x,self.hoop_y])
        player_point = np.array([self.player_x, self.player_y])

        magnitude_r = np.linalg.norm(hoop_point - player_point)

        return magnitude_r
    
    def shouldAddAngle(self) -> bool:
        
        isLeftHoop = False

        isAboveMiddle = False

        if self.hoop_x < 50:

            isLeftHoop = True
        
        if self.player_y > 25:

            isAboveMiddle = True
        
        if ((isLeftHoop) and (not isAboveMiddle)) or ( (not isLeftHoop) and (isAboveMiddle)):
            return False
        else:
            return True
        
        

    def returnDirection(self) -> float:

        shouldAdd : bool = self.shouldAddAngle()
        yOverX = (abs(self.hoop_y - self.player_y)) / (abs(self.hoop_x - self.player_x))

        angle = np.degrees(np.arctan(yOverX))

        direction : float

        if shouldAdd:
            direction = 90 + angle
        else:
            direction =  90 - angle
        
        return direction

    def returnPolarCoordinates(self) -> tuple:
        
        r = self.returnMagnitudeR()

        direction = self.returnDirection()

        return (r,direction)
    
obj = RectangularToPolar(10,20,89,25)

print(obj.returnPolarCoordinates())