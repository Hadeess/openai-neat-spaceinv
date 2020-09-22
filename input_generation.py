import cv2
import numpy as np
import time

# Inputs generated from this function is 76
def inputgen(test_img):

    start = time.time()
    img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    h,w = img.shape     #210, 160

    # Specifics of the game take from screenshots of tests
    # for faster implementation and frame rate
    starting_pixel = 36                     # starting y coor for the aliens
    gap_between_aliens = 18                 # pixel height in between centre f aliens
    horizontal_gap_between_aliens = 16
    down_movement = 10                      # down movement of aliens
    self_y = 192                            # y coor of self
    search_x_start = 22                     # lowest x that aliens can be found
    search_x_end = 139                      # the largest x that aliens can be found
    alien_species = 6                       # no of diffenet aliens
    colour_aliens = 122                     # grayscale colour of the aliens
    colour_self = 98                        # grayscale colour of ourself
    bullet = 142                            # grayscale colour of bullet
    bullet_search = 20                      # keep search limit to [self-bullet_search , self+bullet_search]
    alien_size = 8                          # width of alien

    # Self position
    pos, = np.where(img[self_y]==colour_self)
    if len(pos)==0:
        self_x = 0
    else:
        self_x = pos[(len(pos)//2)]
    
    # Get y coordinate of all aliens
    flag = 1
    iter = 0
    position = []
    vertical_pos = []
    while flag:
        search_index = starting_pixel + 10*iter
        if (search_index) < self_y+10:
            pos, = np.where(img[search_index, search_x_start:search_x_end]==colour_aliens)
        else:
            pos = []
            break
        if (len(pos) == 0):
            iter += 1
        else:
            vertical_pos.append(search_index)
            position.append(pos+22)

            aliens = 0
            while ((aliens+1)*gap_between_aliens+search_index)<self_y+10:
                gap = (aliens+1)*gap_between_aliens
                pos, = np.where(img[search_index+gap,search_x_start:search_x_end]==colour_aliens)
                aliens+=1
                if (len(pos) == 0):
                    iter+=1
                else:
                    vertical_pos.append(search_index+gap)
                    position.append(pos+22)
            flag = 0

    #Extract useful position out of Postion list
    Alien_locations = []
    for i in range(len(position)):
        X = position[i]
        flag = 1
        index = 0
        while flag:
            pos, = np.where( X[index:]<=X[index]+alien_size)
            temp = X[index:][pos]
            alien_coord = (temp[-1]+temp[0])//2
            Alien_locations.append([alien_coord,vertical_pos[i]])
            index = index + len(pos)
            if index>=len(X):
                flag = 0

    #Filling the dead aliens at the end as 0,0
    Enemies_killed = 36-len(Alien_locations)
    for i in range(Enemies_killed):
        Alien_locations.append([0,0])

    Alien_locations = (np.ravel((np.matrix(Alien_locations)).flatten())).tolist()

    #Extract bullet position in vicinity of the self around bullet_search
    search_left = max(self_x-bullet_search,search_x_start)
    search_right = min(self_x + bullet_search,search_x_end)
    bullet_search_area = test_img[starting_pixel:self_y,search_left:search_right]
    #Only consider the closest bullet which is the imminent threat
    temp_bullet = np.argwhere(bullet_search_area==bullet)
    if len(temp_bullet)==0:
        bullet_x,bullet_y = (0,0)
    else:
        nearest_bullet = np.argmax(temp_bullet[:,0])
        bullet_x,bullet_y = temp_bullet[nearest_bullet,1]+search_left,temp_bullet[nearest_bullet,0]+starting_pixel

    #Input to neural nets- self_x,alien_pos_x,alien_pos_y,bullet_x,bullet_y
    Input =  [self_x] + Alien_locations + [bullet_x,bullet_y] + [Enemies_killed]
    end = time.time()

    return Input