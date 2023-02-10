from math import degrees
import numpy as np
from scipy.spatial.transform import Rotation as R

class Cuboid:
    def __init__(self, center, size):
        self.center = center
        self.size = size       
        self.rotation = None
    
    def get_center(self):
        return self.center
    def get_size(self):
        return self.size
    def get_rotation(self):
        return self.rotation
        
    def set_rotation(self, rotation):
        self.rotation = rotation
            
    def get_cuboid_arr(self):

        # center = [x, y, z]
        # size = [l, w, h]

        center = self.get_center()
        size = self.get_size()
        
        l, w, h = size

        step = 0.01
        x_list = np.arange( center[0]-(l/2), center[0]+(l/2), step).tolist()
        y_list = np.arange( center[1]-(w/2), center[1]+(w/2), step).tolist()
        z_list = np.arange( center[2]-(h/2), center[2]+(h/2), step).tolist()
        
        x = [
            [x_list, [center[1]-(w/2) for i in range(len(x_list))], [center[2]-(h/2) for i in range(len(x_list))] ],
            [x_list, [center[1]-(w/2) for i in range(len(x_list))], [center[2]+(h/2) for i in range(len(x_list))] ],
            [x_list, [center[1]+(w/2) for i in range(len(x_list))], [center[2]-(h/2) for i in range(len(x_list))] ],
            [x_list, [center[1]+(w/2) for i in range(len(x_list))], [center[2]+(h/2) for i in range(len(x_list))] ]
            ]
        
        
        y = [
            [ [center[0]-(l/2) for i in range(len(y_list))], y_list, [center[2]-(h/2) for i in range(len(y_list))] ],
            [ [center[0]-(l/2) for i in range(len(y_list))], y_list, [center[2]+(h/2) for i in range(len(y_list))] ],
            [ [center[0]+(l/2) for i in range(len(y_list))], y_list, [center[2]-(h/2) for i in range(len(y_list))] ],
            [ [center[0]+(l/2) for i in range(len(y_list))], y_list, [center[2]+(h/2) for i in range(len(y_list))] ]
            ]
        
        
        z = [
            [ [center[0]-(l/2) for i in range(len(z_list))], [center[1]-(w/2) for i in range(len(z_list))], z_list ],
            [ [center[0]-(l/2) for i in range(len(z_list))], [center[1]+(w/2) for i in range(len(z_list))], z_list ],
            [ [center[0]+(l/2) for i in range(len(z_list))], [center[1]-(w/2) for i in range(len(z_list))], z_list ],
            [ [center[0]+(l/2) for i in range(len(z_list))], [center[1]+(w/2) for i in range(len(z_list))], z_list ]
            ]
        
        x_coord = self.line_to_np(x)
        y_coord = self.line_to_np(y)
        z_coord = self.line_to_np(z)
        if self.get_rotation() is not None:
            x_coord, y_coord, z_coord = self.get_rotate_cube(x_coord, y_coord, z_coord) 
        
        return x_coord, y_coord, z_coord


    def line_to_np(self, lines):
        # num_row = lines.shape[0] * lines.shape[2]
        
        ret = None
        
        temp0 = np.array(lines[0], dtype=np.float32).T
        temp1 = np.array(lines[1], dtype=np.float32).T
        temp2 = np.array(lines[2], dtype=np.float32).T
        temp3 = np.array(lines[3], dtype=np.float32).T
        
        ret = np.concatenate( (temp0, temp1, temp2, temp3), axis=0 )
        
        return ret
    
    def get_rotate_cube(self, x,y,z):
        X, Y, Z = x,y,z
        X = X - self.get_center()
        Y = Y - self.get_center()
        Z = Z - self.get_center()
        rotation = self.get_rotation()  
        temp = R.from_euler('xyz', rotation, degrees=True)
        rmat = np.array(temp.as_matrix())
        
        ret = []
        for pt in X:
            v = rmat @ pt
            ret.append(v)
        X = np.asarray(ret)
        
        ret = []
        for pt in Y:
            v = rmat @ pt
            ret.append(v)
        Y = np.asarray(ret)
        
        ret = []
        for pt in Z:
            v = rmat @ pt
            ret.append(v)
        Z = np.asarray(ret)
        
        X = X + self.get_center()
        Y = Y + self.get_center()
        Z = Z + self.get_center()
        
        return X, Y, Z
        