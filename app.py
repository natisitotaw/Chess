import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk
from chess import Board
import chess
import os
import torch
import torch.nn as nn
import numpy as np
import pickle

class ChessModel(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize the ChessModel with a convolutional neural network architecture.

        Parameters:
        - num_classes (int): Number of output classes for classification.
        """
        super(ChessModel, self).__init__()
        # Define the layers of the network
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)  # Convolutional layer 1
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # Convolutional layer 2
        self.flatten = nn.Flatten()  # Flatten layer
        self.fc1 = nn.Linear(8 * 8 * 128, 256)  # Fully connected layer 1
        self.fc2 = nn.Linear(256, num_classes)  # Fully connected layer 2
        self.relu = nn.ReLU()  # ReLU activation function

        # Initialize weights
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        """
        Define the forward pass of the network.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output logits.
        """
        x = self.relu(self.conv1(x))  # Apply convolutional layer 1 and ReLU
        x = self.relu(self.conv2(x))  # Apply convolutional layer 2 and ReLU
        x = self.flatten(x)  # Flatten the tensor
        x = self.relu(self.fc1(x))  # Apply fully connected layer 1 and ReLU
        x = self.fc2(x)  # Apply fully connected layer 2 to get logits
        return x
    
def board_to_matrix(board: Board):
    """
    Convert a chess board to a matrix representation.

    Parameters:
    - board (Board): Chess board object.

    Returns:
    - np.ndarray: 3D array representing the chess board.
    """
    matrix = np.zeros((13, 8, 8))  # Initialize matrix with 13 channels
    piece_map = board.piece_map()  # Get piece map from the board

    # Populate the matrix with pieces
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    # Populate the legal moves board (13th channel)
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[12, row_to, col_to] = 1

    return matrix

def create_input_for_nn(games):
    """
    Create input data for neural network from a list of games.

    Parameters:
    - games (list): List of chess games.

    Returns:
    - tuple: (X, y) where X is input features and y is target labels.
    """
    X = []
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))  # Append board matrix to input features
            y.append(move.uci())  # Append move in UCI format to labels
            board.push(move)  # Apply the move to the board
    return np.array(X, dtype=np.float32), np.array(y)

def encode_moves(moves):
    """
    Encode chess moves as integers.

    Parameters:
    - moves (list): List of chess moves in UCI format.

    Returns:
    - tuple: (encoded_moves, move_to_int) where encoded_moves are integer representations of moves and move_to_int is a mapping from move to integer.
    """
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return np.array([move_to_int[move] for move in moves], dtype=np.float32), move_to_int

def prepare_input(board: Board):
    """
    Prepare the input tensor for the neural network from a chess board.

    Parameters:
    - board (Board): Chess board object.

    Returns:
    - torch.Tensor: Tensor representation of the board.
    """
    matrix = board_to_matrix(board)
    X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return X_tensor

# Load the move-to-integer mapping
with open("move_to_int", "rb") as file:
    move_to_int = pickle.load(file)

# Determine if a GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


# Load the model
model = ChessModel(num_classes=len(move_to_int))
model_path = 'model.pth'

try:
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.to('cpu')  # Ensure the model is on the CPU
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
model.to(device)
model.eval()  # Set the model to evaluation mode (it may be reductant)

int_to_move = {v: k for k, v in move_to_int.items()}
# Function to make predictions
def predict_move(board: Board):
    X_tensor = prepare_input(board).to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
    
    logits = logits.squeeze(0)  # Remove batch dimension
    
    probabilities = torch.softmax(logits, dim=0).cpu().numpy()  # Convert to probabilities
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = np.argsort(probabilities)[::-1]
    for move_index in sorted_indices:
        move = int_to_move[move_index]
        if move in legal_moves_uci:
            return move
     
    return None


class ChessGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Game")
        
        self.canvas = Canvas(root, width=640, height=640)
        self.canvas.pack()
        
        self.images = {}
        self.load_images()
        
        self.board = self.create_board()
        self.chess_board = Board()
        self.selected_piece = None
        self.current_turn = 'w'  # 'w' for white, 'b' for black
        
        self.draw_board()
        self.canvas.bind("<Button-1>", self.on_click)

        # Add a button to start the game with White's move
        self.start_button = tk.Button(root, text="Start", command=self.start_with_white)
        self.start_button.pack()

    def load_images(self):
        image_paths = {
            'blank': 'Images/60/blank.png',
            'bB': 'Images/bB.png',
            'wB': 'Images/wB.png',
            'bP': 'Images/bP.png',
            'wP': 'Images/wP.png',
            'bN': 'Images/bN.png',
            'wN': 'Images/wN.png',
            'bR': 'Images/bR.png',
            'wR': 'Images/wR.png',
            'bQ': 'Images/bQ.png',
            'wQ': 'Images/wQ.png',
            'bK': 'Images/bK.png',
            'wK': 'Images/wK.png'
        }
        for key, path in image_paths.items():
            image = Image.open(path)
            image = image.resize((80, 80), Image.LANCZOS)
            self.images[key] = ImageTk.PhotoImage(image)

    def create_board(self):
        board = [['blank' for _ in range(8)] for _ in range(8)]
        # Setup initial pieces
        for i in range(8):
            board[1][i] = 'wP'
            board[6][i] = 'bP'
        board[0] = ['wR', 'wN', 'wB', 'wQ', 'wK', 'wB', 'wN', 'wR']
        board[7] = ['bR', 'bN', 'bB', 'bQ', 'bK', 'bB', 'bN', 'bR']
        return board

    def draw_board(self):
        self.canvas.delete("all")
        colors = ['#DDB88C', '#A66D4F']
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                self.canvas.create_rectangle(col*80, row*80, (col+1)*80, (row+1)*80, fill=color)
                piece = self.board[row][col]
                if piece and piece != 'blank':
                    self.canvas.create_image(col*80 + 40, row*80 + 40, image=self.images[piece])
        
        if self.selected_piece:
            self.highlight_moves(self.selected_piece)

    def on_click(self, event):
        col = event.x // 80
        row = event.y // 80

        if self.selected_piece:
            src_row, src_col = self.selected_piece
            if self.is_legal_move(src_row, src_col, row, col):
                if self.current_turn == 'w':
                    self.get_model_move()
                else:
                    self.move_piece(src_row, src_col, row, col)
                    self.current_turn = 'b' if self.current_turn == 'w' else 'w'
                    self.get_model_move()
                
                self.draw_board()
               
            self.selected_piece = None
        else:
            piece = self.board[row][col]
            if piece != 'blank' and piece[0] == self.current_turn:
                self.selected_piece = (row, col)

        self.draw_board()

    def move_piece(self, start_row, start_col, end_row, end_col):
        """ Move a piece on the board and update the board state. """
        piece = self.board[start_row][start_col]
        self.board[end_row][end_col] = piece
        self.board[start_row][start_col] = 'blank'
        
        # Convert board positions to algebraic notation
        start_square = chess.square_name(start_row * 8 + start_col)
        end_square = chess.square_name(end_row * 8 + end_col)
        _temp = start_square+end_square
        print(_temp)
        self.chess_board.push_uci(_temp)
        move = f'{piece} from {start_square} to {end_square}'
        print(move)

    
    def board_to_matrix(self):
        # Convert the tkinter board state to a chess.Board object
        board = Board()
        piece_map = {
            'wP': 'P', 'wR': 'R', 'wN': 'N', 'wB': 'B', 'wQ': 'Q', 'wK': 'K',
            'bP': 'p', 'bR': 'r', 'bN': 'n', 'bB': 'b', 'bQ': 'q', 'bK': 'k'
        }
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece != 'blank':
                    board.set_piece_at(8 * row + col, chess.Piece.from_symbol(piece_map[piece]))
        return board


    def board_to_chess_board(self):
        board = chess.Board()
        piece_map = {
            'wP': 'P', 'wR': 'R', 'wN': 'N', 'wB': 'B', 'wQ': 'Q', 'wK': 'K',
            'bP': 'p', 'bR': 'r', 'bN': 'n', 'bB': 'b', 'bQ': 'q', 'bK': 'k'
        }
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece != 'blank':
                    board.set_piece_at(8 * row + col, chess.Piece.from_symbol(piece_map[piece]))
                    print(piece, 8 * row , col)
        return board
    
    def get_model_move(self):
        chess_board = self.chess_board
        if self.current_turn == 'b':
            print("It's white's turn. Model does not move.")
            return
        else:
            best_move = predict_move(chess_board)  # Assuming predict_move takes a chess.Board
           
            if best_move:
                self.move_piece_from_uci(best_move)
                self.current_turn = 'b'  # Switch back to black's turn after model plays

    def move_piece_from_uci(self, move_uci):
        chess_board = self.chess_board  # Create a chess.Board object
        move = chess.Move.from_uci(move_uci)
        if move in chess_board.legal_moves:
            self.move_piece(move.from_square // 8, move.from_square % 8, move.to_square // 8, move.to_square % 8)
            self.draw_board() 

    def is_legal_move(self, start_row, start_col, end_row, end_col):
        piece = self.board[start_row][start_col]
        if piece == 'blank':
            return False
        
        # Check if the destination is occupied by a piece of the same color
        destination_piece = self.board[end_row][end_col]
        if destination_piece != 'blank' and destination_piece[0] == piece[0]:
            return False

        # Check if the move is valid for the piece type
        if piece == 'wP' or piece == 'bP':
            # Pawn movement
            direction = 1 if piece == 'wP' else -1
            if start_col == end_col:
                if (end_row == start_row + direction and self.board[end_row][end_col] == 'blank') or \
                (end_row == start_row + 2 * direction and start_row == (1 if piece == 'wP' else 6) and self.board[end_row][end_col] == 'blank'):
                    return True
            elif abs(start_col - end_col) == 1 and end_row == start_row + direction and destination_piece != 'blank':
                return True

        elif piece == 'wR' or piece == 'bR':
            # Rook movement
            if start_row == end_row or start_col == end_col:
                if self.is_path_clear(start_row, start_col, end_row, end_col):
                    return True

        elif piece == 'wN' or piece == 'bN':
            # Knight movement
            if (abs(start_row - end_row) == 2 and abs(start_col - end_col) == 1) or \
            (abs(start_row - end_row) == 1 and abs(start_col - end_col) == 2):
                return True

        elif piece == 'wB' or piece == 'bB':
            # Bishop movement
            if abs(start_row - end_row) == abs(start_col - end_col):
                if self.is_path_clear(start_row, start_col, end_row, end_col):
                    return True

        elif piece == 'wQ' or piece == 'bQ':
            # Queen movement
            if start_row == end_row or start_col == end_col or abs(start_row - end_row) == abs(start_col - end_col):
                if self.is_path_clear(start_row, start_col, end_row, end_col):
                    return True

        return False


    def is_path_clear(self, start_row, start_col, end_row, end_col):
        """ Check if the path from start to end is clear. """
        if start_row == end_row:  # Horizontal movement
            step = 1 if start_col < end_col else -1
            for col in range(start_col + step, end_col, step):
                if self.board[start_row][col] != 'blank':
                    return False
        elif start_col == end_col:  # Vertical movement
            step = 1 if start_row < end_row else -1
            for row in range(start_row + step, end_row, step):
                if self.board[row][start_col] != 'blank':
                    return False
        elif abs(start_row - end_row) == abs(start_col - end_col):  # Diagonal movement
            row_step = 1 if start_row < end_row else -1
            col_step = 1 if start_col < end_col else -1
            row, col = start_row + row_step, start_col + col_step
            while row != end_row:
                if self.board[row][col] != 'blank':
                    return False
                row += row_step
                col += col_step
        return True
    def start_with_white(self):
        # Set the initial turn to White and redraw the board
        self.current_turn = 'w'
        self.get_model_move()
        self.draw_board()

    def highlight_moves(self, piece_pos):
        """ Highlight all possible moves for the selected piece. """
        start_row, start_col = piece_pos
        piece = self.board[start_row][start_col]
        if piece == 'blank':
            return
        
        move_color = '#098524'  # Green color for highlighting
        
        def is_within_bounds(r, c):
            return 0 <= r < 8 and 0 <= c < 8
        
        dot_radius = 20  # Radius of the dot
        
        def draw_highlight_dot(row, col):
            """ Draw a green dot at the specified position. """
            x0 = col * 80 + (80 - dot_radius) / 2
            y0 = row * 80 + (80 - dot_radius) / 2
            x1 = x0 + dot_radius
            y1 = y0 + dot_radius
            self.canvas.create_oval(x0, y0, x1, y1, fill=move_color, outline=move_color)
        
        if piece == 'wP' or piece == 'bP':
            # Pawn possible moves
            direction = 1 if piece == 'wP' else -1
            # Forward move
            if is_within_bounds(start_row + direction, start_col) and self.board[start_row + direction][start_col] == 'blank':
                draw_highlight_dot(start_row + direction, start_col)
            # Diagonal capture
            if is_within_bounds(start_row + direction, start_col - 1) and self.board[start_row + direction][start_col - 1] != 'blank':
                draw_highlight_dot(start_row + direction, start_col - 1)
            if is_within_bounds(start_row + direction, start_col + 1) and self.board[start_row + direction][start_col + 1] != 'blank':
                draw_highlight_dot(start_row + direction, start_col + 1)
        
        elif piece == 'wR' or piece == 'bR':
            # Rook possible moves
            for r in range(start_row + 1, 8):
                if is_within_bounds(r, start_col):
                    if self.is_legal_move(start_row, start_col, r, start_col):
                        draw_highlight_dot(r, start_col)
                    if self.board[r][start_col] != 'blank':
                        break
                else:
                    break
            for r in range(start_row - 1, -1, -1):
                if is_within_bounds(r, start_col):
                    if self.is_legal_move(start_row, start_col, r, start_col):
                        draw_highlight_dot(r, start_col)
                    if self.board[r][start_col] != 'blank':
                        break
                else:
                    break
            for c in range(start_col + 1, 8):
                if is_within_bounds(start_row, c):
                    if self.is_legal_move(start_row, start_col, start_row, c):
                        draw_highlight_dot(start_row, c)
                    if self.board[start_row][c] != 'blank':
                        break
                else:
                    break
            for c in range(start_col - 1, -1, -1):
                if is_within_bounds(start_row, c):
                    if self.is_legal_move(start_row, start_col, start_row, c):
                        draw_highlight_dot(start_row, c)
                    if self.board[start_row][c] != 'blank':
                        break
                else:
                    break

        elif piece == 'wN' or piece == 'bN':
            # Knight possible moves
            for dr, dc in [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]:
                new_row, new_col = start_row + dr, start_col + dc
                if is_within_bounds(new_row, new_col) and self.is_legal_move(start_row, start_col, new_row, new_col):
                    draw_highlight_dot(new_row, new_col)
        
        elif piece == 'wB' or piece == 'bB':
            # Bishop possible moves
            for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                r, c = start_row, start_col
                while True:
                    r += dr
                    c += dc
                    if is_within_bounds(r, c):
                        if self.is_legal_move(start_row, start_col, r, c):
                            draw_highlight_dot(r, c)
                        if self.board[r][c] != 'blank':
                            break
                    else:
                        break
        
        elif piece == 'wQ' or piece == 'bQ':
            # Queen possible moves (combination of rook and bishop)
            # Rook moves
            for r in range(start_row + 1, 8):
                if is_within_bounds(r, start_col):
                    if self.is_legal_move(start_row, start_col, r, start_col):
                        draw_highlight_dot(r, start_col)
                    if self.board[r][start_col] != 'blank':
                        break
                else:
                    break
            for r in range(start_row - 1, -1, -1):
                if is_within_bounds(r, start_col):
                    if self.is_legal_move(start_row, start_col, r, start_col):
                        draw_highlight_dot(r, start_col)
                    if self.board[r][start_col] != 'blank':
                        break
                else:
                    break
            for c in range(start_col + 1, 8):
                if is_within_bounds(start_row, c):
                    if self.is_legal_move(start_row, start_col, start_row, c):
                        draw_highlight_dot(start_row, c)
                    if self.board[start_row][c] != 'blank':
                        break
                else:
                    break
            for c in range(start_col - 1, -1, -1):
                if is_within_bounds(start_row, c):
                    if self.is_legal_move(start_row, start_col, start_row, c):
                        draw_highlight_dot(start_row, c)
                    if self.board[start_row][c] != 'blank':
                        break
                else:
                    break
            # Bishop moves
            for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                r, c = start_row, start_col
                while True:
                    r += dr
                    c += dc
                    if is_within_bounds(r, c):
                        if self.is_legal_move(start_row, start_col, r, c):
                            draw_highlight_dot(r, c)
                        if self.board[r][c] != 'blank':
                            break
                    else:
                        break


if __name__ == "__main__":
    root = tk.Tk() 
    game = ChessGame(root)
    root.mainloop()
