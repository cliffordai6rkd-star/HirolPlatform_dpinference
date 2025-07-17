from math import inf as infinity
from random import choice
import platform
import time
from os import system
import numpy as np
HUMAN = -1
COMP = 1

choice = {
        HUMAN: ' ',
        COMP: ' ',
    }

"""
An implementation of Minimax AI Algorithm in Tic Tac Toe,
using Python.
This software is available under GPL license.
Author: Clederson Cruz
Year: 2017
License: GNU GENERAL PUBLIC LICENSE (GPL)
"""


def evaluate(state):
    """
    Function to heuristic evaluation of state.
    :param state: the state of the current board
    :return: +1 if the computer wins; -1 if the human wins; 0 draw
    """
    if wins(state, COMP):
        score = +1
    elif wins(state, HUMAN):
        score = -1
    else:
        score = 0

    return score

def map_to_board(holes, red_chesses, white_chesses):
    """
    根据检测结果生成棋盘状态
    参数:
        holes (list): 洞的位置列表
        red_chesses (list): 红棋的位置列表
        white_chesses (list): 白棋的位置列表
    返回:
        board (list): 棋盘状态
    """

    # 只处理状态为on的对象
    all_objects = [
        *[obj for obj in holes if (obj[-1] == 'on')],
        #or (obj[-1] == 'off') or (obj[-1] == 'unknown')
        *[obj for obj in red_chesses if (obj[-1] == 'on')],
        #or (obj[-1] == 'off') or (obj[-1] == 'unknown')
        *[obj for obj in white_chesses if (obj[-1] == 'on')] 
        #or (obj[-1] == 'off') or (obj[-1] == 'unknown')
    ]

    if len(all_objects) == 9:
        # # 获取坐标极值
        # x_coords = [obj[0] for obj in all_objects]
        # y_coords = [obj[1] for obj in all_objects]

        # # 确定四个角点
        # left = min(x_coords)
        # bottom = min(y_coords)
        # top = max(y_coords)
        # right = max(x_coords)
        
        # # 找到 obj_right
        # obj_left = None
        # obj_top = None
        # obj_bottom = None
        # obj_right = None
        # for obj in all_objects:
        #     if obj[0] == left:
        #         obj_left = obj
        #     if obj[1] == top:
        #         obj_top = obj
        #     if obj[1] == bottom:
        #         obj_bottom = obj
        #     if obj[0] == right:
        #         obj_right = obj

        # 生成所有两两组合并计算距离
        from itertools import combinations
        # 将列表对象转换为元组
        tuple_objects = [tuple(obj) for obj in all_objects]
        pairs = list(combinations(tuple_objects, 2))
        
        # 计算所有点对的欧氏距离（使用前两个坐标）
        distances = [
            (p1, p2, np.linalg.norm(
                np.array([float(p1[0]), float(p1[1])]) - 
                np.array([float(p2[0]), float(p2[1])])
            ))
            for p1, p2 in pairs
        ]
        
        # 按距离降序排序，取前两个最大的不重叠对角线
        distances.sort(key=lambda x: -x[2])
        
        # 选择两组最大距离且不共享端点的对角线
        diag1_p1, diag1_p2, _ = distances[0]
        diag2_p1, diag2_p2 = None, None
        for d in distances[1:]:
            if d[0] not in (diag1_p1, diag1_p2) and d[1] not in (diag1_p1, diag1_p2):
                diag2_p1, diag2_p2, _ = d
                break
        
        # 确定四个候选角点（使用元组坐标）
        candidate_points = {diag1_p1, diag1_p2, diag2_p1, diag2_p2}

        # 找到原始对象对应的坐标点
        sorted_points = sorted(candidate_points, key=lambda p: p[0])
        min_two_x_points = sorted_points[:2]
        # 取 p[0] 前二小中，p[1] 最小的点
        obj_left = min(min_two_x_points, key=lambda p: p[1])
        obj_top = max(min_two_x_points, key=lambda p: p[1])
        top_two_x_points = sorted_points[-2:]
        # 取 p[0] 前二小中，p[1] 最大的点
        obj_right = max(top_two_x_points, key=lambda p: p[1])
        obj_bottom = min(top_two_x_points, key=lambda p: p[1])

        # obj_left = min(candidate_points, key=lambda p: p[0])
        # obj_right = max(candidate_points, key=lambda p: p[0])
        # obj_top = max(candidate_points, key=lambda p: p[1])
        # obj_bottom = min(candidate_points, key=lambda p: p[1])

        x_base = [(a - b)/2 for a, b in zip(obj_top[0:2], obj_left[0:2])]
        y_base = [(a - b)/2 for a, b in zip(obj_bottom[0:2], obj_left[0:2])]

        # 生成棋盘状态
        board = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        for obj in all_objects:
            # 计算相对于左上角点的偏移量
            dx = obj[0] - obj_left[0]
            dy = obj[1] - obj_left[1]

            # 使用 numpy 数组进行点乘运算
            x_base = np.array(x_base)
            y_base = np.array(y_base)
            offset = np.array([dx, dy])

            # 计算在 x_base 和 y_base 方向上的投影
            x_proj = np.dot(offset, x_base) / np.dot(x_base, x_base)
            y_proj = np.dot(offset, y_base) / np.dot(y_base, y_base)

            # 映射到棋盘的 3x3 网格中
            x = int((x_proj+0.3) // 1)
            y = int((y_proj+0.3) // 1)

            # 确保 x 和 y 在棋盘范围内
            x = max(0, min(2, x))
            y = max(0, min(2, y))
            
            board[y][x] = obj[-2]

        return board, obj_left, x_base, y_base
    else:
        return None, None, None, None

def wins(state, player):
    """
    This function tests if a specific player wins. Possibilities:
    * Three rows    [X X X] or [O O O]
    * Three cols    [X X X] or [O O O]
    * Two diagonals [X X X] or [O O O]
    :param state: the state of the current board
    :param player: a human or a computer
    :return: True if the player wins
    """
    win_state = [
        [state[0][0], state[0][1], state[0][2]],
        [state[1][0], state[1][1], state[1][2]],
        [state[2][0], state[2][1], state[2][2]],
        [state[0][0], state[1][0], state[2][0]],
        [state[0][1], state[1][1], state[2][1]],
        [state[0][2], state[1][2], state[2][2]],
        [state[0][0], state[1][1], state[2][2]],
        [state[2][0], state[1][1], state[0][2]],
    ]
    if [choice[player], choice[player], choice[player]] in win_state:
        return True
    else:
        return False

def game_over(state):
    """
    This function test if the human or computer wins
    :param state: the state of the current board
    :return: True if the human or computer wins
    """
    return wins(state, HUMAN) or wins(state, COMP)

def empty_cells(state):
    """
    Each empty cell will be added into cells' list
    :param state: the state of the current board
    :return: a list of empty cells
    """
    cells = []

    for x, row in enumerate(state):
        for y, cell in enumerate(row):
            if cell == 'hole':
                cells.append([x, y])
                
    return cells

def valid_move(x, y):
    """
    A move is valid if the chosen cell is empty
    :param x: X coordinate
    :param y: Y coordinate
    :return: True if the board[x][y] is empty
    """
    if [x, y] in empty_cells(board):
        return True
    else:
        return False

def set_move(x, y, player):
    """
    Set the move on board, if the coordinates are valid
    :param x: X coordinate
    :param y: Y coordinate
    :param player: the current player
    """
    if valid_move(x, y):
        board[x][y] = player
        return True
    else:
        return False

def minimax(state, depth, player):
    """
    AI function that choice the best move
    :param state: current state of the board
    :param depth: node index in the tree (0 <= depth <= 9),
    but never nine in this case (see iaturn() function)
    :param player: an human or a computer
    :return: a list with [the best row, best col, best score]
    """
    if player == COMP:
        best = [-1, -1, -infinity]
    else:
        best = [-1, -1, +infinity]

    if depth == 0 or game_over(state):
        score = evaluate(state)
        return [-1, -1, score]

    for cell in empty_cells(state):
        x, y = cell[0], cell[1]
        state[x][y] = choice[player]
        score = minimax(state, depth - 1, -player)
        state[x][y] = 'hole'
        score[0], score[1] = x, y

        if player == COMP:
            if score[2] > best[2]:
                best = score  # max value
        else:
            if score[2] < best[2]:
                best = score  # min value

    return best

def render(state, c_choice, h_choice):
    """
    Print the board on console
    :param state: current state of the board
    """

    chars = {
        -1: h_choice,
        +1: c_choice,
        0: ' '
    }
    str_line = '---------------'

    print('\n' + str_line)
    for row in state:
        for cell in row:
            symbol = chars[cell]
            print(f'| {symbol} |', end='')
        print('\n' + str_line)

def ai_turn(c_choice, h_choice, board):
    """
    It calls the minimax function if the depth < 9,
    else it choices a random coordinate.
    :param c_choice: computer's choice X or O
    :param h_choice: human's choice X or O
    :return: (x, y) coordinates of the move
    """
    depth = len(empty_cells(board))
    if depth == 0 or game_over(board):
        return None
    if depth == 9:
        x = choice([0, 1, 2])
        y = choice([0, 1, 2])
    else:
        move = minimax(board, depth, COMP)
        x, y = move[0], move[1]
    return x, y

def chessplan(holes, red_chesses, white_chesses):
    board,obj_left, x_base, y_base = map_to_board(holes, red_chesses, white_chesses)
    print(board)
    # 棋盘状态统计逻辑
    white_count = sum(row.count('white_chess') for row in board) if board else 0
    red_count = sum(row.count('red_chess') for row in board) if board else 0
    
    if red_count == white_count +1 :
        h_choice = 'red_chess'
        c_choice = 'white_chess'
        loca_chess = find_chess(white_chesses)
        choice[HUMAN] = 'red_chess'
        choice[COMP] = 'white_chess'
    elif red_count == white_count -1:
        h_choice = 'white_chess'
        c_choice = 'red_chess'
        choice[HUMAN] = 'white_chess'
        choice[COMP] = 'red_chess'
        loca_chess = find_chess(red_chesses)
    else:
        print("The number of red and white chesses is confused. Please wait.")
        return None,None,None
    
    if loca_chess is None:
        print("No chess to move.")
        return None,None,None
    print(loca_chess)
    # Main loop of this game
    if len(empty_cells(board)) > 0 and not game_over(board):
        move = ai_turn(c_choice, h_choice, board)
        print(move)
        for hole in holes:
            # 计算相对于左上角点的偏移量
            dx = hole[0] - obj_left[0]
            dy = hole[1] - obj_left[1]

            # 使用 numpy 数组进行点乘运算
            x_base = np.array(x_base)
            y_base = np.array(y_base)
            offset = np.array([dx, dy])

            # 计算在 x_base 和 y_base 方向上的投影
            x_proj = np.dot(offset, x_base) / np.dot(x_base, x_base)
            y_proj = np.dot(offset, y_base) / np.dot(y_base, y_base)

            # 映射到棋盘的 3x3 网格中
            x = int((x_proj+0.3) // 1)
            y = int((y_proj+0.3) // 1)
            if x == move[1] and y == move[0]:
                closest_hole = hole[:2]
        # loca_move = [
        #     obj_left[0] + move[0] * x_base[0] + move[0] * y_base[0],
        #     obj_left[1] + move[1] * x_base[1] + move[1] * y_base[1]
        # ]
        # # 找到所有hole的位置
        # hole_locations = []
        # for hole in holes:
        #     if hole[-1] == 'on':
        #         hole_locations.append(hole[:2])
    
        # # 计算每个hole到loca_move的距离
        # distances = []
        # for hole_location in hole_locations:
        #     distance = np.linalg.norm(np.array(hole_location) - np.array(loca_move))
        #     distances.append(distance)
        
        # # 找到最近的hole
        # if distances:
        #     closest_hole_index = np.argmin(distances)
        #     closest_hole = hole_locations[closest_hole_index]
        # else:
        #     closest_hole = None
        return closest_hole, move, loca_chess
    
    else:
        return None,None,None

    # # Game over message
    # if wins(board, HUMAN):
    #     print(f'Human turn [{h_choice}]')
    #     render(board, c_choice, h_choice)
    #     return 'YOU WIN!'
    # elif wins(board, COMP):
    #     print(f'Computer turn [{c_choice}]')
    #     render(board, c_choice, h_choice)
    #     return 'YOU LOSE!'
    # else:
    #     render(board, c_choice, h_choice)
    #     return 'DRAW!'

def find_chess(chesses):
    """
    找到没下的棋子
    """
    for chess in chesses:
        if chess[-1] == 'off' :
            return chess[:2]
    print("No chess to move.")
    return None    
    